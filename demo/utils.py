from tqdm import tqdm
from datasets import load_dataset
import pickle as pkl
import numpy as np
import time
from scipy.optimize import minimize

### Utility functions
def sigmoid(z):
    return 1/(1+np.exp(-z))

def item_curve(theta, a, b):
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)

def fit_theta(responses_test, seen_items, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]
    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(responses_test[seen_items] * np.log(P + eps) + (1 - responses_test[seen_items]) * np.log(1 - P + eps))
        return -log_likelihood
    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, np.zeros(D), method = optimizer).x[None,:,None] 
    return optimal_theta

### Evaluation function
def evaluate(model_correctness, 
             anchor_points, 
             unseen_items, 
             IRT_parameters, 
             number_of_examples):
                 
    A, B, weights, lambd, N, balance_weights = IRT_parameters
    
    theta = fit_theta(model_correctness, anchor_points, A, B)
    data_part_IRTp = ((balance_weights*model_correctness)[anchor_points]).mean()
    irt_part = (balance_weights*item_curve(theta.reshape(1, A.shape[1], 1), A, B))[0, [unseen_items]].mean()
    IRTp_lambd = number_of_examples/N
    
    # Getting estimates
    IRT = (weights*model_correctness[anchor_points]).sum()
    IRTp = IRTp_lambd * data_part_IRTp + (1 - IRTp_lambd) * irt_part
    IRTpp = lambd*IRT + (1-lambd)*IRTp
    
    return IRT, IRTp, IRTpp


### Data preprocessing
def get_IRT_model_and_tinyMMLU(sample_data: dict, 
                               method: str, 
                               number_of_examples: int):

    method = 'anchor-irt' if method == 'irt' else 'anchor' if method == 'correct.' else None
    assert method is not None
    
    it = {'anchor':0, 'anchor-irt':4}[method]
    weights = sample_data['item_weights'][method][number_of_examples][it]['mmlu']
    anchor_points = sample_data['seen_items'][method][number_of_examples][it]
    lambd = sample_data['opt_lambds'][method+'_gpirt']['mmlu'][number_of_examples]
    scenarios_position = sample_data['scenarios_position']['mmlu']
    subscenarios_position = sample_data['subscenarios_position']['mmlu']
    balance_weights = np.ones(len(scenarios_position))
    N = len(scenarios_position)
    n_sub = len(subscenarios_position)
    
    for sub in subscenarios_position.keys():
        n_i = len(subscenarios_position[sub])
    balance_weights[subscenarios_position[sub]] = N/(n_sub*n_i) 
    
    it = {'anchor':0, 'anchor-irt':4}[method]
    
    weights = sample_data['item_weights'][method][number_of_examples][it]['mmlu']
    anchor_points = sample_data['seen_items'][method][number_of_examples][it]
    lambd = sample_data['opt_lambds'][method+'_gpirt']['mmlu'][number_of_examples]
    
    A = sample_data['A']
    B = sample_data['B']
    
    scenarios_position = sample_data['scenarios_position']['mmlu']
    subscenarios_position = sample_data['subscenarios_position']['mmlu']
    N = len(scenarios_position)
    balance_weights = np.ones(N)
    n_sub = len(subscenarios_position)
    for sub in subscenarios_position.keys():
        n_i = len(subscenarios_position[sub])
        balance_weights[subscenarios_position[sub]] = N/(n_sub*n_i) 
    unseen_items = [i for i in range(N) if i not in anchor_points]
    
    return anchor_points, unseen_items, (A, B, weights, lambd, N, balance_weights)
    

def download_and_preprocess_correctness(models):
    
    mmlu_correctness_raw = download_model_correctness(models)
    mmlu_correctness_preprocessed = preprocess_model_correctness(mmlu_correctness_raw)
    responses_stacked = prepare_responses(mmlu_correctness_preprocessed)
    
    return mmlu_correctness_preprocessed, responses_stacked

def prepare_responses(data):
    """ Stack all responses of different subscenarios """
    responses = [np.vstack([data['data'][sub]['correctness'] for sub in mmlu_subscenarios]).T for scenario in ['mmlu']]
    return np.hstack(responses)

def preprocess_model_correctness(mmlu_correctness_raw = None):
    if mmlu_correctness_raw is None:
        with open('mmlu.pickle', 'rb') as handle:
            mmlu_correctness_raw = pkl.load(handle)
    
    
    df = mmlu_correctness_raw
    models = list(df.keys())
    
    for key1 in df.keys():
        for key2 in df[key1].keys():
            if df[key1][key2]==None: 
                try: models.remove(key1)
                except: pass
                
    data = {}
    data['data'] = {}
    data['models'] = [models]
    
    for sub in df[list(df.keys())[0]].keys():
        data['data'][sub] = {}
        data['data'][sub]['correctness'] = []
        
        for model in models:
            data['data'][sub]['correctness'].append(df[model][sub]['correctness'])
                
        data['data'][sub]['correctness'] = np.array(data['data'][sub]['correctness']).T.astype(float)
        

    with open('mmlu_processed.pickle', 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    return data


def download_model_correctness(models):
    data = {}
    for model in tqdm(models):
        data[model] = {}
        for s in mmlu_subscenarios:
            data[model][s] = {}
            data[model][s]['correctness'] = None
            data[model][s]['dates'] = None
            
    skipped = 0
    log = []
    for model in tqdm(models):
        skipped_aux=0
        for s in mmlu_subscenarios:
            if 'arc' in s: metric = 'acc_norm'
            elif 'hellaswag' in s: metric = 'acc_norm'
            elif 'truthfulqa' in s: metric = 'mc2'
            else: metric = 'acc'
    
            try:
                aux = load_dataset(model, s)
                data[model][s]['dates'] = list(aux.keys())
                data[model][s]['correctness'] = [a[metric] for a in aux['latest']['metrics']]
                #print("\nOK {:} {:}\n".format(model,s))
                log.append("\nOK {:} {:}\n".format(model,s))
            except Exception as e:
                print("\n")
                print(f"An error occurred: {e}")
                print("Trying to get the data using a different strategy...")
                try:
                    aux = load_dataset(model, s)
                    data[model][s]['dates'] = list(aux.keys())
                    data[model][s]['correctness'] = aux['latest'][metric]
                    #print("\nOK {:} {:}\n".format(model,s))
                    log.append("\nOK {:} {:}\n".format(model,s))
                except Exception as e:
                    print(f"An error occurred: {e}")
                    data[model][s] = None
                    skipped_aux+=1
                    log.append("\nSKIP {:} {:}\n".format(model,s))

        if skipped_aux>0: skipped+=1
            
        with open('mmlu.pickle', 'wb') as handle:
            pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
    print("\nModels skipped in total: {:}\n".format(skipped))
    print("\n Correctness scores saved to mmlu.pickle")
    
    return data


mmlu_subscenarios = ['harness_hendrycksTest_abstract_algebra_5', 'harness_hendrycksTest_anatomy_5', 
                     'harness_hendrycksTest_astronomy_5', 'harness_hendrycksTest_business_ethics_5', 
                     'harness_hendrycksTest_clinical_knowledge_5', 'harness_hendrycksTest_college_biology_5', 
                     'harness_hendrycksTest_college_chemistry_5', 'harness_hendrycksTest_college_computer_science_5', 
                     'harness_hendrycksTest_college_mathematics_5', 'harness_hendrycksTest_college_medicine_5', 
                     'harness_hendrycksTest_college_physics_5', 'harness_hendrycksTest_computer_security_5', 
                     'harness_hendrycksTest_conceptual_physics_5', 'harness_hendrycksTest_econometrics_5', 
                     'harness_hendrycksTest_electrical_engineering_5', 'harness_hendrycksTest_elementary_mathematics_5', 
                     'harness_hendrycksTest_formal_logic_5', 'harness_hendrycksTest_global_facts_5', 
                     'harness_hendrycksTest_high_school_biology_5', 'harness_hendrycksTest_high_school_chemistry_5', 
                     'harness_hendrycksTest_high_school_computer_science_5', 'harness_hendrycksTest_high_school_european_history_5', 
                     'harness_hendrycksTest_high_school_geography_5', 'harness_hendrycksTest_high_school_government_and_politics_5', 
                     'harness_hendrycksTest_high_school_macroeconomics_5', 'harness_hendrycksTest_high_school_mathematics_5', 
                     'harness_hendrycksTest_high_school_microeconomics_5', 'harness_hendrycksTest_high_school_physics_5', 
                     'harness_hendrycksTest_high_school_psychology_5', 'harness_hendrycksTest_high_school_statistics_5',
                     'harness_hendrycksTest_high_school_us_history_5', 'harness_hendrycksTest_high_school_world_history_5', 
                     'harness_hendrycksTest_human_aging_5', 'harness_hendrycksTest_human_sexuality_5', 
                     'harness_hendrycksTest_international_law_5', 'harness_hendrycksTest_jurisprudence_5', 
                     'harness_hendrycksTest_logical_fallacies_5', 'harness_hendrycksTest_machine_learning_5', 
                     'harness_hendrycksTest_management_5', 'harness_hendrycksTest_marketing_5', 
                     'harness_hendrycksTest_medical_genetics_5', 'harness_hendrycksTest_miscellaneous_5',
                     'harness_hendrycksTest_moral_disputes_5', 'harness_hendrycksTest_moral_scenarios_5', 
                     'harness_hendrycksTest_nutrition_5', 'harness_hendrycksTest_philosophy_5', 
                     'harness_hendrycksTest_prehistory_5', 'harness_hendrycksTest_professional_accounting_5',
                     'harness_hendrycksTest_professional_law_5', 'harness_hendrycksTest_professional_medicine_5', 
                     'harness_hendrycksTest_professional_psychology_5', 'harness_hendrycksTest_public_relations_5',
                     'harness_hendrycksTest_security_studies_5', 'harness_hendrycksTest_sociology_5', 
                     'harness_hendrycksTest_us_foreign_policy_5', 'harness_hendrycksTest_virology_5', 
                     'harness_hendrycksTest_world_religions_5']


available_models = ['open-llm-leaderboard/details_moreh__MoMo-70B-lora-1.8.6-DPO',
                     'open-llm-leaderboard/details_cloudyu__Yi-34Bx3-MoE-90B',
                     'open-llm-leaderboard/details_Weyaxi__Helion-4x34B',
                     'open-llm-leaderboard/details_Weyaxi__Bagel-Hermes-34B-Slerp',
                     'open-llm-leaderboard/details_Weyaxi__Bagel-Hermes-2x34b',
                     'open-llm-leaderboard/details_nfaheem__Marcoroni-7b-DPO-Merge',
                     'open-llm-leaderboard/details_alnrg2arg__test2_3',
                     'open-llm-leaderboard/details_jondurbin__bagel-dpo-34b-v0.2',
                     'open-llm-leaderboard/details_udkai__Turdus',
                     'open-llm-leaderboard/details_jondurbin__bagel-dpo-34b-v0.2',
                     'open-llm-leaderboard/details_gagan3012__MetaModel_moe',
                     'open-llm-leaderboard/details_jeonsworld__CarbonVillain-en-10.7B-v3',
                     'open-llm-leaderboard/details_TomGrc__FusionNet',
                     'open-llm-leaderboard/details_kekmodel__StopCarbon-10.7B-v6',
                     'open-llm-leaderboard/details_jeonsworld__CarbonVillain-en-10.7B-v1',
                     'open-llm-leaderboard/details_Weyaxi__SauerkrautLM-UNA-SOLAR-Instruct',
                     'open-llm-leaderboard/details_VAGOsolutions__SauerkrautLM-SOLAR-Instruct',
                     'open-llm-leaderboard/details_bhavinjawade__SOLAR-10B-Nector-DPO-Jawade',
                     'open-llm-leaderboard/details_kyujinpy__Sakura-SOLAR-Instruct-DPO-v2',
                     'open-llm-leaderboard/details_fblgit__UNA-SOLAR-10.7B-Instruct-v1.0',
                     'open-llm-leaderboard/details_kyujinpy__Sakura-SOLRCA-Instruct-DPO',
                     'open-llm-leaderboard/details_zhengr__MixTAO-7Bx2-MoE-DPO',
                     'open-llm-leaderboard/details_Weyaxi__Nous-Hermes-2-SUS-Chat-2x34B',
                     'open-llm-leaderboard/details_NousResearch__Nous-Hermes-2-Yi-34B',
                     'open-llm-leaderboard/details_flemmingmiguel__NeuDist-Ro-7B',
                     'open-llm-leaderboard/details_mlabonne__NeuralMarcoro14-7B',
                     'open-llm-leaderboard/details_cookinai__BruinHermes',
                     'open-llm-leaderboard/details_shadowml__Daredevil-7B',
                     'open-llm-leaderboard/details_zyh3826__GML-Mistral-merged-v1',
                     'open-llm-leaderboard/details_Sao10K__WinterGoddess-1.4x-70B-L2',
                     'open-llm-leaderboard/details_CultriX__MistralTrixTest',
                     'open-llm-leaderboard/details_rombodawg__Open_Gpt4_8x7B',
                     'open-llm-leaderboard/details_shadowml__Marcoro14-7B-ties',
                     'open-llm-leaderboard/details_VAGOsolutions__SauerkrautLM-Mixtral-8x7B-Instruct',
                     'open-llm-leaderboard/details_PSanni__MPOMixtral-8x7B-Instruct-v0.1',
                     'open-llm-leaderboard/details_VAGOsolutions__SauerkrautLM-Mixtral-8x7B-Instruct',
                     'open-llm-leaderboard/details_maywell__PiVoT-SUS-RP',
                     'open-llm-leaderboard/details_rwitz2__pee',
                     'open-llm-leaderboard/details_Brillibits__Instruct_Mixtral-8x7B-v0.1_Dolly15K',
                     'open-llm-leaderboard/details_mindy-labs__mindy-7b',
                     'open-llm-leaderboard/details_janhq__supermario-slerp',
                     'open-llm-leaderboard/details_rishiraj__CatPPT-base',
                     'open-llm-leaderboard/details_SanjiWatsuki__Kunoichi-7B',
                     'open-llm-leaderboard/details_NousResearch__Nous-Hermes-2-Mixtral-8x7B-SFT',
                     'open-llm-leaderboard/details_brucethemoose__Yi-34B-200K-DARE-merge-v5',
                     'open-llm-leaderboard/details_AA051611__A0110',
                     'open-llm-leaderboard/details_Weyaxi__openchat-3.5-1210-Seraph-Slerp',
                     'open-llm-leaderboard/details_Weyaxi__openchat-3.5-1210-Seraph-Slerp',
                     'open-llm-leaderboard/details_SanjiWatsuki__Loyal-Macaroni-Maid-7B',
                     'open-llm-leaderboard/details_AA051610__A0106',
                     'open-llm-leaderboard/details_PulsarAI__OpenHermes-2.5-neural-chat-v3-3-Slerp',
                     'open-llm-leaderboard/details_Walmart-the-bag__Solar-10.7B-Cato',
                     'open-llm-leaderboard/details_Weyaxi__OpenHermes-2.5-neural-chat-v3-3-openchat-3.5-1210-Slerp',
                     'open-llm-leaderboard/details_Intel__neural-chat-7b-v3-3-Slerp',
                     'open-llm-leaderboard/details_KnutJaegersberg__Deacon-34b-Adapter',
                     'open-llm-leaderboard/details_TomGrc__FusionNet_SOLAR',
                     'open-llm-leaderboard/details_superlazycoder__NeuralPipe-7B-slerp',
                     'open-llm-leaderboard/details_NousResearch__Nous-Hermes-2-SOLAR-10.7B',
                     'open-llm-leaderboard/details_chanwit__flux-7b-v0.1',
                     'open-llm-leaderboard/details_one-man-army__una-neural-chat-v3-3-P2-OMA',
                     'open-llm-leaderboard/details_Q-bert__MetaMath-Cybertron',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.2',
                     'open-llm-leaderboard/details_perlthoughts__Chupacabra-8x7B-MoE',
                     'open-llm-leaderboard/details_perlthoughts__Falkor-7b',
                     'open-llm-leaderboard/details_APMIC__caigun-lora-model-34B-v3',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.5-LASER-0.1',
                     'open-llm-leaderboard/details_rishiraj__oswald-7b',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.4',
                     'open-llm-leaderboard/details_flemmingmiguel__Distilled-HermesChat-7B',
                     'open-llm-leaderboard/details_Weyaxi__MetaMath-OpenHermes-2.5-neural-chat-v3-3-Slerp',
                     'open-llm-leaderboard/details_Intel__neural-chat-7b-v3-3',
                     'open-llm-leaderboard/details_internlm__internlm2-20b',
                     'open-llm-leaderboard/details_migtissera__Tess-M-v1.3',
                     'open-llm-leaderboard/details_fblgit__una-cybertron-7b-v2-bf16',
                     'open-llm-leaderboard/details_chargoddard__mixtralmerge-8x7B-rebalanced-test',
                     'open-llm-leaderboard/details_FelixChao__WizardDolphin-7B',
                     'open-llm-leaderboard/details_FelixChao__ExtremeDolphin-MoE',
                     'open-llm-leaderboard/details_rishiraj__oswald-2x7b',
                     'open-llm-leaderboard/details_Sao10K__Sensualize-Mixtral-bf16',
                     'open-llm-leaderboard/details_OpenBuddy__openbuddy-deepseek-67b-v15-base',
                     'open-llm-leaderboard/details_garage-bAInd__Platypus2-70B-instruct',
                     'open-llm-leaderboard/details_jondurbin__airoboros-l2-70b-2.2.1',
                     'open-llm-leaderboard/details_diffnamehard__Mistral-CatMacaroni-slerp-gradient',
                     'open-llm-leaderboard/details_chargoddard__servile-harpsichord-cdpo',
                     'open-llm-leaderboard/details_sethuiyer__distilabled_Chikuma_10.7B',
                     'open-llm-leaderboard/details_AIDC-ai-business__Marcoroni-70B-v1',
                     'open-llm-leaderboard/details_AA051611__limb',
                     'open-llm-leaderboard/details_adamo1139__Yi-34B-AEZAKMI-v1',
                     'open-llm-leaderboard/details_jondurbin__spicyboros-70b-2.2',
                     'open-llm-leaderboard/details_psmathur__model_009',
                     'open-llm-leaderboard/details_mistralai__Mixtral-8x7B-v0.1',
                     'open-llm-leaderboard/details_mistralai__Mixtral-8x7B-v0.1',
                     'open-llm-leaderboard/details_kyujinpy__PlatYi-34B-Llama',
                     'open-llm-leaderboard/details_nlpguy__ColorShadow-7B',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.5-LASER-0.4',
                     'open-llm-leaderboard/details_decapoda-research__Antares-11b-v1',
                     'open-llm-leaderboard/details_Sao10K__Sensualize-Solar-10.7B',
                     'open-llm-leaderboard/details_LoSboccacc__orthogonal-2x7B-base',
                     'open-llm-leaderboard/details_Azazelle__xDAN-SlimOrca',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.5-LASER-exp2-0.1',
                     'open-llm-leaderboard/details_kyujinpy__PlatYi-34B-200k-Q-FastChat',
                     'open-llm-leaderboard/details_Swisslex__Mixtral-Orca-v0.1',
                     'open-llm-leaderboard/details_RatanRohith__MistralBeagle-RS-7B-V0.1',
                     'open-llm-leaderboard/details_mrfakename__NeuralOrca-7B-v1',
                     'open-llm-leaderboard/details_openaccess-ai-collective__DPOpenHermes-7B',
                     'open-llm-leaderboard/details_bongchoi__MoMo-70B-LoRA-V1.1',
                     'open-llm-leaderboard/details_fblgit__juanako-7b-UNA',
                     'open-llm-leaderboard/details_Praneeth__StarMix-7B-slerp',
                     'open-llm-leaderboard/details_charlesdedampierre__TopicNeuralHermes-2.5-Mistral-7B',
                     'open-llm-leaderboard/details_diffnamehard__Mistral-CatMacaroni-slerp-uncensored',
                     'open-llm-leaderboard/details_beberik__rawr',
                     'open-llm-leaderboard/details_macadeliccc__laser-dolphin-mixtral-2x7b-dpo',
                     'open-llm-leaderboard/details_perlthoughts__Starling-LM-alpha-8x7B-MoE',
                     'open-llm-leaderboard/details_perlthoughts__Chupacabra-7B-v2',
                     'open-llm-leaderboard/details_Open-Orca__Mixtral-SlimOrca-8x7B',
                     'open-llm-leaderboard/details_macadeliccc__polyglot-math-4x7b',
                     'open-llm-leaderboard/details_Sao10K__Frostwind-10.7B-v1',
                     'open-llm-leaderboard/details_Mihaiii__Pallas-0.5-LASER-0.6',
                     'open-llm-leaderboard/details_Yhyu13__LMCocktail-Mistral-7B-v1',
                     'open-llm-leaderboard/details_rombodawg__Leaderboard-killer-MoE_4x7b',
                     'open-llm-leaderboard/details_Brillibits__Instruct_Llama70B_Dolly15k',
                     'open-llm-leaderboard/details_augtoma__qCammel-70x',
                     'open-llm-leaderboard/details_Doctor-Shotgun__mythospice-limarp-70b',
                     'open-llm-leaderboard/details_chargoddard__mistral-11b-slimorca',
                     'open-llm-leaderboard/details_TokenBender__pic_7B_mistral_Full_v0.1',
                     'open-llm-leaderboard/details_TomGrc__FusionNet_passthrough',
                     'open-llm-leaderboard/details_perlthoughts__Chupacabra-7B-v2.03-128k',
                     'open-llm-leaderboard/details_TomGrc__FusionNet_passthrough_v0.1',
                     'open-llm-leaderboard/details_notbdq__alooowso',
                     'open-llm-leaderboard/details_Delcos__Velara-11B-V2',
                     'open-llm-leaderboard/details_jondurbin__bagel-7b-v0.1',
                     'open-llm-leaderboard/details_Mihaiii__Metis-0.3',
                     'open-llm-leaderboard/details_perlthoughts__Chupacabra-7B-v2.03',
                     'open-llm-leaderboard/details_SanjiWatsuki__neural-chat-7b-v3-3-wizardmath-dare-me',
                     'open-llm-leaderboard/details_simonveitner__MathHermes-2.5-Mistral-7B',
                     'open-llm-leaderboard/details_cognitivecomputations__dolphin-2.2.1-mistral-7b',
                     'open-llm-leaderboard/details_bn22__OpenHermes-2.5-Mistral-7B-MISALIGNED',
                     'open-llm-leaderboard/details_jarradh__llama2_70b_chat_uncensored',
                     'open-llm-leaderboard/details_Sao10K__Euryale-L2-70B',
                     'open-llm-leaderboard/details_jondurbin__airoboros-l2-70b-gpt4-m2.0',
                     'open-llm-leaderboard/details_upstage__llama-65b-instruct',
                     'open-llm-leaderboard/details_OpenAssistant__llama2-70b-oasst-sft-v10',
                     'open-llm-leaderboard/details_elinas__chronos-70b-v2',
                     'open-llm-leaderboard/details_Neuronovo__neuronovo-7B-v0.1',
                     'open-llm-leaderboard/details_openbmb__UltraLM-65b',
                     'open-llm-leaderboard/details_jae24__openhermes_dpo_norobot_0201',
                     'open-llm-leaderboard/details_monology__openinstruct-mistral-7b',
                     'open-llm-leaderboard/details_KaeriJenti__Kaori-34B-v1',
                     'open-llm-leaderboard/details_argilla__notus-7b-v1',
                     'open-llm-leaderboard/details_xxyyy123__Mistral7B_adaptor_v1',
                     'open-llm-leaderboard/details_xDAN-AI__xDAN-L1Mix-DeepThinking-v2',
                     'open-llm-leaderboard/details_liuda1__dm7b_sft_gpt88w_merge',
                     'open-llm-leaderboard/details_KnutJaegersberg__Qwen-14B-Llamafied',
                     'open-llm-leaderboard/details_HenryJJ__dolphin-2.6-mistral-7b-dpo-orca-v3',
                     'open-llm-leaderboard/details_UCLA-AGI__test',
                     'open-llm-leaderboard/details_huggyllama__llama-65b',
                     'open-llm-leaderboard/details_sr5434__CodegebraGPT-10b',
                     'open-llm-leaderboard/details_upaya07__Birbal-7B-V1',
                     'open-llm-leaderboard/details_migtissera__Tess-XS-v1-3-yarn-128K',
                     'open-llm-leaderboard/details_UCLA-AGI__test0',
                     'open-llm-leaderboard/details_Azazelle__Half-NSFW_Noromaid-7b',
                     'open-llm-leaderboard/details_migtissera__Tess-7B-v1.4',
                     'open-llm-leaderboard/details_kyujinpy__PlatYi-34B-200K-Q',
                     'open-llm-leaderboard/details_chargoddard__MelangeC-70b',
                     'open-llm-leaderboard/details_spmurrayzzz__Mistral-Syndicate-7B',
                     'open-llm-leaderboard/details_dfurman__Mistral-7B-Instruct-v0.2',
                     'open-llm-leaderboard/details_huangyt__Mistral-7B-v0.1-Open-Platypus_2.5w-r16-gate_up_down',
                     'open-llm-leaderboard/details_Intel__neural-chat-7b-v3-1',
                     'open-llm-leaderboard/details_tianlinliu0121__zephyr-7b-dpo-full-beta-0.2',
                     'open-llm-leaderboard/details_TheBloke__robin-65b-v2-fp16',
                     'open-llm-leaderboard/details_uukuguy__speechless-llama2-13b',
                     'open-llm-leaderboard/details_microsoft__phi-2',
                     'open-llm-leaderboard/details_WizardLM__WizardLM-70B-V1.0',
                     'open-llm-leaderboard/details_huggingface__llama-65b',
                     'open-llm-leaderboard/details_kyujinpy__PlatYi-34B-Llama-Q-v3',
                     'open-llm-leaderboard/details_Dans-DiscountModels__Dans-07YahooAnswers-7b',
                     'open-llm-leaderboard/details_OpenBuddy__openbuddy-falcon-40b-v16.1-4k',
                     'open-llm-leaderboard/details_HiTZ__alpaca-lora-65b-en-pt-es-ca',
                     'open-llm-leaderboard/details_OpenBuddyEA__openbuddy-llama-30b-v7.1-bf16',
                     'open-llm-leaderboard/details_OpenBuddyEA__openbuddy-llama-30b-v7.1-bf16',
                     'open-llm-leaderboard/details_Sao10K__Zephyrus-L1-33B',
                     'open-llm-leaderboard/details_acrastt__kalomaze-stuff',
                     'open-llm-leaderboard/details_HenryJJ__Instruct_Mistral-7B-v0.1_Dolly15K',
                     'open-llm-leaderboard/details_speechlessai__speechless-mistral-7b-dare-0.85',
                     'open-llm-leaderboard/details_diffnamehard__Psyfighter2-Noromaid-ties-Capybara-13B',
                     'open-llm-leaderboard/details_vihangd__smartyplats-7b-v2',
                     'open-llm-leaderboard/details_CallComply__SOLAR-10.7B-Instruct-v1.0-128k',
                     'open-llm-leaderboard/details_teknium__CollectiveCognition-v1-Mistral-7B',
                     'open-llm-leaderboard/details_Mihaiii__Metis-0.1',
                     'open-llm-leaderboard/details_CallComply__Starling-LM-11B-alpha',
                     'open-llm-leaderboard/details_jilp00__Hermes-2-SOLAR-10.7B-Symbolic',
                     'open-llm-leaderboard/details_crumb__apricot-wildflower-20',
                     'open-llm-leaderboard/details_Locutusque__Orca-2-13B-no_robots',
                     'open-llm-leaderboard/details_maywell__Synatra-RP-Orca-2-7b-v0.1',
                     'open-llm-leaderboard/details_HuggingFaceH4__zephyr-7b-alpha',
                     'open-llm-leaderboard/details_hywu__Camelidae-8x13B',
                     'open-llm-leaderboard/details_migtissera__SynthIA-7B-v1.3',
                     'open-llm-leaderboard/details_SuperAGI__SAM',
                     'open-llm-leaderboard/details_maywell__Synatra-7B-v0.3-RP',
                     'open-llm-leaderboard/details_bofenghuang__vigostral-7b-chat',
                     'open-llm-leaderboard/details_abdulrahman-nuzha__finetuned-Mistral-5000-v1.0',
                     'open-llm-leaderboard/details_lilloukas__Platypus-30B',
                     'open-llm-leaderboard/details_osanseviero__mistral-instruct-frankenmerge',
                     'open-llm-leaderboard/details_akjindal53244__Mistral-7B-v0.1-Open-Platypus',
                     'open-llm-leaderboard/details_uukuguy__speechless-code-mistral-7b-v1.0',
                     'open-llm-leaderboard/details_ajibawa-2023__scarlett-33b',
                     'open-llm-leaderboard/details_jondurbin__airoboros-m-7b-3.1.2',
                     'open-llm-leaderboard/details_Aeala__GPT4-x-AlpacaDente2-30b',
                     'open-llm-leaderboard/details_PeanutJar__Mistral-v0.1-PeanutButter-v0.0.2-7B',
                     'open-llm-leaderboard/details_CobraMamba__mamba-gpt-7b-v1',
                     'open-llm-leaderboard/details_umd-zhou-lab__claude2-alpaca-13B',
                     'open-llm-leaderboard/details_Undi95__MLewd-ReMM-L2-Chat-20B',
                     'open-llm-leaderboard/details_Aspik101__trurl-2-13b-pl-instruct_unload',
                     'open-llm-leaderboard/details_ajibawa-2023__Uncensored-Frank-33B',
                     'open-llm-leaderboard/details_Aspik101__30B-Lazarus-instruct-PL-lora_unload',
                     'open-llm-leaderboard/details_lgaalves__mistral-7b-platypus1k',
                     'open-llm-leaderboard/details_martyn__llama-megamerge-dare-13b',
                     'open-llm-leaderboard/details_Sao10K__Stheno-1.8-L2-13B',
                     'open-llm-leaderboard/details_tiiuae__falcon-40b',
                     'open-llm-leaderboard/details_Undi95__Mistral-11B-v0.1',
                     'open-llm-leaderboard/details_martyn__llama2-megamerge-dare-13b-v2',
                     'open-llm-leaderboard/details_oh-yeontaek__llama-2-13B-LoRA-assemble',
                     'open-llm-leaderboard/details_Aeala__VicUnlocked-alpaca-30b',
                     'open-llm-leaderboard/details_Sao10K__Stheno-v2-Delta-fp16',
                     'open-llm-leaderboard/details_JosephusCheung__Pwen-14B-Chat-20_30',
                     'open-llm-leaderboard/details_Zangs3011__mistral_7b_DolphinCoder',
                     'open-llm-leaderboard/details_l3utterfly__mistral-7b-v0.1-layla-v1',
                     'open-llm-leaderboard/details_alignment-handbook__zephyr-7b-sft-full',
                     'open-llm-leaderboard/details_PocketDoc__Dans-AdventurousWinds-7b',
                     'open-llm-leaderboard/details_SkunkworksAI__Mistralic-7B-1',
                     'open-llm-leaderboard/details_Sao10K__BrainDerp2',
                     'open-llm-leaderboard/details_PulsarAI__2x-LoRA-Assemble-Nova-13B',
                     'open-llm-leaderboard/details_Undi95__MLewd-Chat-v2-13B',
                     'open-llm-leaderboard/details_jondurbin__airoboros-33b-gpt4-m2.0',
                     'open-llm-leaderboard/details_Undi95__ReMM-v2.2-L2-13B',
                     'open-llm-leaderboard/details_stabilityai__StableBeluga-13B',
                     'open-llm-leaderboard/details_WebraftAI__synapsellm-7b-mistral-v0.3-preview',
                     'open-llm-leaderboard/details_TheBloke__OpenOrca-Platypus2-13B-GPTQ',
                     'open-llm-leaderboard/details_huggingface__llama-30b',
                     'open-llm-leaderboard/details_Undi95__Emerald-13B',
                     'open-llm-leaderboard/details_TIGER-Lab__TIGERScore-13B',
                     'open-llm-leaderboard/details_Undi95__ReMM-v2.1-L2-13B',
                     'open-llm-leaderboard/details_chargoddard__storytime-13b',
                     'open-llm-leaderboard/details_BELLE-2__BELLE-Llama2-13B-chat-0.4M',
                     'open-llm-leaderboard/details_Brouz__Slerpeno',
                     'open-llm-leaderboard/details_sauce1337__BerrySauce-L2-13b',
                     'open-llm-leaderboard/details_PulsarAI__EnsembleV5-Nova-13B',
                     'open-llm-leaderboard/details_SciPhi__SciPhi-Self-RAG-Mistral-7B-32k',
                     'open-llm-leaderboard/details_Sao10K__Stheno-L2-13B',
                     'open-llm-leaderboard/details_uukuguy__speechless-code-mistral-7b-v2.0',
                     'open-llm-leaderboard/details_Gryphe__MythoMix-L2-13b',
                     'open-llm-leaderboard/details_Aspik101__StableBeluga-13B-instruct-PL-lora_unload',
                     'open-llm-leaderboard/details_Locutusque__Orca-2-13b-SFT-v6',
                     'open-llm-leaderboard/details_Austism__chronos-hermes-13b-v2',
                     'open-llm-leaderboard/details_The-Face-Of-Goonery__Huginn-13b-v1.2',
                     'open-llm-leaderboard/details_The-Face-Of-Goonery__huginnv1.2',
                     'open-llm-leaderboard/details_Undi95__Nous-Hermes-13B-Code',
                     'open-llm-leaderboard/details_migtissera__Synthia-13B-v1.2',
                     'open-llm-leaderboard/details_YeungNLP__firefly-llama2-13b-v1.2',
                     'open-llm-leaderboard/details_Danielbrdz__Barcenas-13b',
                     'open-llm-leaderboard/details_lu-vae__llama2-13B-sharegpt4-orca-openplatypus-8w',
                     'open-llm-leaderboard/details_meta-llama__Llama-2-13b-hf',
                     'open-llm-leaderboard/details_Undi95__U-Amethyst-20B',
                     'open-llm-leaderboard/details_defog__sqlcoder-34b-alpha',
                     'open-llm-leaderboard/details_uukuguy__speechless-hermes-coig-lite-13b',
                     'open-llm-leaderboard/details_kingbri__chronolima-airo-grad-l2-13B',
                     'open-llm-leaderboard/details_Expert68__llama2_13b_instructed_version2',
                     'open-llm-leaderboard/details_mosaicml__mpt-30b-chat',
                     'open-llm-leaderboard/details_TFLai__Luban-Platypus2-13B-QLora-0.80-epoch',
                     'open-llm-leaderboard/details_ewqr2130__mistral-se-inst-ppo',
                     'open-llm-leaderboard/details_elinas__chronos-13b-v2',
                     'open-llm-leaderboard/details_Aspik101__vicuna-13b-v1.5-PL-lora_unload',
                     'open-llm-leaderboard/details_Sao10K__Mythical-Destroyer-V2-L2-13B',
                     'open-llm-leaderboard/details_jondurbin__airoboros-c34b-2.2.1',
                     'open-llm-leaderboard/details_PygmalionAI__pygmalion-2-13b',
                     'open-llm-leaderboard/details_speechlessai__speechless-llama2-dolphin-orca-platypus-13b',
                     'open-llm-leaderboard/details_ajibawa-2023__Python-Code-33B',
                     'open-llm-leaderboard/details_duliadotio__dulia-13b-8k-alpha',
                     'open-llm-leaderboard/details_lmsys__vicuna-13b-v1.5-16k',
                     'open-llm-leaderboard/details_WebraftAI__synapsellm-7b-mistral-v0.4-preview3',
                     'open-llm-leaderboard/details_digitous__13B-Chimera',
                     'open-llm-leaderboard/details_The-Face-Of-Goonery__Huginn-13b-FP16',
                     'open-llm-leaderboard/details_openaccess-ai-collective__manticore-13b',
                     'open-llm-leaderboard/details_ehartford__Samantha-1.11-CodeLlama-34b',
                     'open-llm-leaderboard/details_The-Face-Of-Goonery__Chronos-Beluga-v2-13bfp16',
                     'open-llm-leaderboard/details_elyza__ELYZA-japanese-Llama-2-13b-instruct',
                     'open-llm-leaderboard/details_Secbone__llama-2-13B-instructed',
                     'open-llm-leaderboard/details_TFLai__Nous-Hermes-Platypus2-13B-QLoRA-0.80-epoch',
                     'open-llm-leaderboard/details_KoboldAI__LLaMA2-13B-Holomax',
                     'open-llm-leaderboard/details_BAAI__Aquila2-34B',
                     'open-llm-leaderboard/details_CallComply__zephyr-7b-beta-128k',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-huangyt_Fintune_1_17w-q_k_v_o_proj',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE3_3.3w-r16-gate_up_down',
                     'open-llm-leaderboard/details_openaccess-ai-collective__wizard-mega-13b',
                     'open-llm-leaderboard/details_totally-not-an-llm__EverythingLM-13b-V3-peft',
                     'open-llm-leaderboard/details_budecosystem__genz-13b-v2',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE1_17w-r4',
                     'open-llm-leaderboard/details_TheBloke__Wizard-Vicuna-13B-Uncensored-HF',
                     'open-llm-leaderboard/details_hfl__chinese-alpaca-2-13b-16k',
                     'open-llm-leaderboard/details_yeontaek__llama-2-13b-Beluga-QLoRA',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE4_3.8w-r8-q_k_v_o',
                     'open-llm-leaderboard/details_TheBloke__airoboros-13B-HF',
                     'open-llm-leaderboard/details_jondurbin__airoboros-13b',
                     'open-llm-leaderboard/details_ehartford__based-30b',
                     'open-llm-leaderboard/details_WizardLM__WizardMath-13B-V1.0',
                     'open-llm-leaderboard/details_Weyaxi__Platypus-Nebula-v2-7B',
                     'open-llm-leaderboard/details_euclaise__Ferret-7B',
                     'open-llm-leaderboard/details_yeontaek__llama-2-13b-QLoRA',
                     'open-llm-leaderboard/details_zyh3826__llama2-13b-ft-openllm-leaderboard-v1',
                     'open-llm-leaderboard/details_Envoid__Libra-19B',
                     'open-llm-leaderboard/details_NobodyExistsOnTheInternet__GiftedConvo13bLoraNoEconsE4',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE3_3.3w-r8-gate_up_down',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-huangyt_FINETUNE2_3w',
                     'open-llm-leaderboard/details_shareAI__bimoGPT-llama2-13b',
                     'open-llm-leaderboard/details_chargoddard__llama2-22b-blocktriangular',
                     'open-llm-leaderboard/details_KnutJaegersberg__deacon-13b',
                     'open-llm-leaderboard/details_NobodyExistsOnTheInternet__PuffedConvo13bLoraE4',
                     'open-llm-leaderboard/details_ehartford__WizardLM-1.0-Uncensored-CodeLlama-34b',
                     'open-llm-leaderboard/details_circulus__Llama-2-7b-orca-v1',
                     'open-llm-leaderboard/details_chargoddard__llama2-22b-blocktriangular',
                     'open-llm-leaderboard/details_IGeniusDev__llama13B-quant8-testv1-openorca-customdataset',
                     'open-llm-leaderboard/details_psmathur__orca_mini_v3_7b',
                     'open-llm-leaderboard/details_TigerResearch__tigerbot-13b-base',
                     'open-llm-leaderboard/details_Aeala__GPT4-x-Alpasta-13b',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE5_4w-r4-q_k_v_o',
                     'open-llm-leaderboard/details_kevinpro__Vicuna-13B-CoT',
                     'open-llm-leaderboard/details_AdaptLLM__finance-chat',
                     'open-llm-leaderboard/details_meta-math__MetaMath-Llemma-7B',
                     'open-llm-leaderboard/details_TFLai__Airboros2.1-Platypus2-13B-QLora-0.80-epoch',
                     'open-llm-leaderboard/details_Undi95__MLewd-L2-13B',
                     'open-llm-leaderboard/details_wei123602__llama2-13b-FINETUNE3_TEST',
                     'open-llm-leaderboard/details_KnutJaegersberg__Walter-Mistral-7B',
                     'open-llm-leaderboard/details_xzuyn__Alpacino-SuperCOT-13B',
                     'open-llm-leaderboard/details_dhmeltzer__Llama-2-13b-hf-ds_wiki_1024_full_r_64_alpha_16_merged',
                     'open-llm-leaderboard/details_pe-nlp__llama-2-13b-platypus-vicuna-wizard',
                     'open-llm-leaderboard/details_CHIH-HUNG__llama-2-13b-FINETUNE4_3.8w-r8-q_k_v_o_gate_up_down',
                     'open-llm-leaderboard/details_Undi95__Llama2-13B-no_robots-alpaca-lora',
                     'open-llm-leaderboard/details_totally-not-an-llm__EverythingLM-13b-V2-16k',
                     'open-llm-leaderboard/details_PocketDoc__Dans-PersonalityEngine-13b',
                     'open-llm-leaderboard/details_HyperbeeAI__Tulpar-7b-v0',
                     'open-llm-leaderboard/details_teknium__Mistral-Trismegistus-7B',
                     'open-llm-leaderboard/details_heegyu__LIMA-13b-hf',
                     'open-llm-leaderboard/details_wahaha1987__llama_13b_sharegpt94k_fastchat',
                     'open-llm-leaderboard/details_lvkaokao__llama2-7b-hf-chat-lora-v2',
                     'open-llm-leaderboard/details_wang7776__Llama-2-7b-chat-hf-10-sparsity',
                     'open-llm-leaderboard/details_camel-ai__CAMEL-13B-Combined-Data',
                     'open-llm-leaderboard/details_Unbabel__TowerInstruct-7B-v0.1',
                     'open-llm-leaderboard/details_beaugogh__Llama2-7b-openorca-mc-v2-dpo',
                     'open-llm-leaderboard/details_mncai__Llama2-7B-guanaco-1k',
                     'open-llm-leaderboard/details_beaugogh__Llama2-7b-openorca-mc-v1',
                     'open-llm-leaderboard/details_HyperbeeAI__Tulpar-7b-v1',
                     'open-llm-leaderboard/details_OpenBuddy__openbuddy-mixtral-7bx8-v16.3-32k',
                     'open-llm-leaderboard/details_LTC-AI-Labs__L2-7b-Beluga-WVG-Test',
                     'open-llm-leaderboard/details_lmsys__vicuna-7b-v1.5',
                     'open-llm-leaderboard/details_pe-nlp__llama-2-13b-vicuna-wizard',
                     'open-llm-leaderboard/details_ashercn97__manatee-7b',
                     'open-llm-leaderboard/details_umd-zhou-lab__recycled-wizardlm-7b-v2.0',
                     'open-llm-leaderboard/details_jphme__em_german_leo_mistral',
                     'open-llm-leaderboard/details_rombodawg__LosslessMegaCoder-llama2-7b-mini',
                     'open-llm-leaderboard/details_abhinand__tamil-llama-13b-instruct-v0.1',
                     'open-llm-leaderboard/details_jondurbin__airoboros-c34b-2.1',
                     'open-llm-leaderboard/details_JosephusCheung__Pwen-VL-Chat-20_30',
                     'open-llm-leaderboard/details_camel-ai__CAMEL-13B-Role-Playing-Data',
                     'open-llm-leaderboard/details_Open-Orca__OpenOrca-Preview1-13B',
                     'open-llm-leaderboard/details_zarakiquemparte__kuchiki-l2-7b',
                     'open-llm-leaderboard/details_Locutusque__Rhino-Mistral-7B',
                     'open-llm-leaderboard/details_LTC-AI-Labs__L2-7b-Synthia-WVG-Test',
                     'open-llm-leaderboard/details_TheBloke__koala-13B-HF',
                     'open-llm-leaderboard/details_PygmalionAI__pygmalion-2-7b',
                     'open-llm-leaderboard/details_deepseek-ai__deepseek-moe-16b-base',
                     'open-llm-leaderboard/details_ziqingyang__chinese-llama-2-13b',
                     'open-llm-leaderboard/details_AlekseyKorshuk__vic15-exp-syn-fight-cp3838',
                     'open-llm-leaderboard/details_davzoku__cria-llama2-7b-v1.3',
                     'open-llm-leaderboard/details_YeungNLP__firefly-llama2-13b-pretrain',
                     'open-llm-leaderboard/details_wang7776__Mistral-7B-Instruct-v0.2-sparsity-20',
                     'open-llm-leaderboard/details_DopeorNope__LaOT',
                     'open-llm-leaderboard/details_maximuslee07__llama-2-7b-rockwell-final',
                     'open-llm-leaderboard/details_922-CA__monika-ddlc-7b-v1',
                     'open-llm-leaderboard/details_openthaigpt__openthaigpt-1.0.0-beta-13b-chat-hf',
                     'open-llm-leaderboard/details_NewstaR__Koss-7B-chat',
                     'open-llm-leaderboard/details_Charlie911__vicuna-7b-v1.5-lora-timedial',
                     'open-llm-leaderboard/details_TheBloke__tulu-7B-fp16',
                     'open-llm-leaderboard/details_kashif__stack-llama-2',
                     'open-llm-leaderboard/details_haoranxu__ALMA-13B',
                     'open-llm-leaderboard/details_togethercomputer__Llama-2-7B-32K-Instruct',
                     'open-llm-leaderboard/details_RoversX__llama-2-7b-hf-small-shards-Samantha-V1-SFT',
                     'open-llm-leaderboard/details_llm-agents__tora-code-34b-v1.0',
                     'open-llm-leaderboard/details_TinyPixel__testmodel2',
                     'open-llm-leaderboard/details_WizardLM__WizardMath-7B-V1.0',
                     'open-llm-leaderboard/details_Charlie911__vicuna-7b-v1.5-lora-mixed-datasets-time-unit',
                     'open-llm-leaderboard/details_bongchoi__test-llama2-7b',
                     'open-llm-leaderboard/details_TaylorAI__Flash-Llama-7B',
                     'open-llm-leaderboard/details_luffycodes__vicuna-class-shishya-ac-hal-13b-ep3',
                     'open-llm-leaderboard/details_HuggingFaceH4__starchat-beta',
                     'open-llm-leaderboard/details_clibrain__Llama-2-7b-ft-instruct-es',
                     'open-llm-leaderboard/details_dotvignesh__perry-7b',
                     'open-llm-leaderboard/details_quantumaikr__QuantumLM-7B',
                     'open-llm-leaderboard/details_ceadar-ie__FinanceConnect-13B',
                     'open-llm-leaderboard/details_heegyu__LIMA2-7b-hf',
                     'open-llm-leaderboard/details_PocketDoc__Dans-RetroRodeo-13b',
                     'open-llm-leaderboard/details_ajibawa-2023__scarlett-7b',
                     'open-llm-leaderboard/details_martyn__mistral-megamerge-dare-7b',
                     'open-llm-leaderboard/details_dhmeltzer__llama-7b-SFT_ds_wiki65k_1024_r_64_alpha_16_merged',
                     'open-llm-leaderboard/details_GOAT-AI__GOAT-7B-Community',
                     'open-llm-leaderboard/details_jondurbin__airoboros-7b-gpt4-1.1',
                     'open-llm-leaderboard/details_llm-agents__tora-7b-v1.0',
                     'open-llm-leaderboard/details_cognitivecomputations__yayi2-30b-llama',
                     'open-llm-leaderboard/details_AlpinDale__pygmalion-instruct',
                     'open-llm-leaderboard/details_TheBloke__Wizard-Vicuna-7B-Uncensored-HF',
                     'open-llm-leaderboard/details_jondurbin__airoboros-l2-7b-gpt4-m2.0',
                     'open-llm-leaderboard/details_DevaMalla__llama_7b_qlora_pds-eval',
                     'open-llm-leaderboard/details_webbigdata__ALMA-7B-Ja-V2',
                     'open-llm-leaderboard/details_bofenghuang__vigogne-7b-instruct',
                     'open-llm-leaderboard/details_microsoft__phi-1_5',
                     'open-llm-leaderboard/details_golaxy__gowizardlm',
                     'open-llm-leaderboard/details_Neko-Institute-of-Science__metharme-7b',
                     'open-llm-leaderboard/details_jondurbin__airoboros-7b',
                     'open-llm-leaderboard/details_h2m__mhm-7b-v1.3',
                     'open-llm-leaderboard/details_Undi95__Mixtral-8x7B-MoE-RP-Story',
                     'open-llm-leaderboard/details_itsliupeng__openllama-7b-base',
                     'open-llm-leaderboard/details_ausboss__llama7b-wizardlm-unfiltered',
                     'open-llm-leaderboard/details_DevaMalla__llama_7b_lora',
                     'open-llm-leaderboard/details_ehartford__dolphin-2.2-yi-34b-200k',
                     'open-llm-leaderboard/details_stabilityai__stablelm-3b-4e1t',
                     'open-llm-leaderboard/details_cognitivecomputations__dolphin-2.2-yi-34b-200k',
                     'open-llm-leaderboard/details_jondurbin__airoboros-7b-gpt4-1.4.1-qlora',
                     'open-llm-leaderboard/details_YeungNLP__firefly-llama2-7b-pretrain',
                     'open-llm-leaderboard/details_fireballoon__baichuan-vicuna-chinese-7b',
                     'open-llm-leaderboard/details_vikash06__mistral_v1',
                     'open-llm-leaderboard/details_huggingface__llama-7b',
                     'open-llm-leaderboard/details_yeontaek__WizardCoder-Python-13B-LoRa',
                     'open-llm-leaderboard/details_Charlie911__vicuna-7b-v1.5-lora-mctaco-modified1',
                     'open-llm-leaderboard/details_ashercn97__giraffe-7b',
                     'open-llm-leaderboard/details_luffycodes__llama-shishya-7b-ep3-v1',
                     'open-llm-leaderboard/details_speechlessai__speechless-coding-7b-16k-tora',
                     'open-llm-leaderboard/details_shareAI__CodeLLaMA-chat-13b-Chinese',
                     'open-llm-leaderboard/details_KnutJaegersberg__Qwen-1_8B-Llamafied',
                     'open-llm-leaderboard/details_WeOpenML__Alpaca-7B-v1',
                     'open-llm-leaderboard/details_mosaicml__mpt-7b',
                     'open-llm-leaderboard/details_togethercomputer__GPT-JT-6B-v0',
                     'open-llm-leaderboard/details_hyunseoki__ko-ref-llama2-13b',
                     'open-llm-leaderboard/details_cyberagent__calm2-7b-chat',
                     'open-llm-leaderboard/details_FreedomIntelligence__phoenix-inst-chat-7b',
                     'open-llm-leaderboard/details_Pierre-obi__Mistral_solar-slerp',
                     'open-llm-leaderboard/details_qblocks__codellama_7b_DolphinCoder',
                     'open-llm-leaderboard/details_openlm-research__open_llama_7b',
                     'open-llm-leaderboard/details_klosax__open_llama_13b_600bt_preview',
                     'open-llm-leaderboard/details_AlekseyKorshuk__pygmalion-6b-vicuna-chatml',
                     'open-llm-leaderboard/details_wenge-research__yayi-7b',
                     'open-llm-leaderboard/details_glaiveai__glaive-coder-7b',
                     'open-llm-leaderboard/details_uukuguy__speechless-coder-ds-6.7b',
                     'open-llm-leaderboard/details_digitous__Javalion-R',
                     'open-llm-leaderboard/details_codellama__CodeLlama-34b-Python-hf',
                     'open-llm-leaderboard/details_NousResearch__CodeLlama-34b-hf',
                     'open-llm-leaderboard/details_codellama__CodeLlama-7b-hf',
                     'open-llm-leaderboard/details_heegyu__RedTulu-Uncensored-3B-0719',
                     'open-llm-leaderboard/details_GeorgiaTechResearchInstitute__galactica-6.7b-evol-instruct-70k',
                     'open-llm-leaderboard/details_HuggingFaceH4__starchat-alpha']


