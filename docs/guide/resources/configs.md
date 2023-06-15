Each labeling run with the autolabel library requires a config to be specified. The config has 5 top-level keys and several nested keys, many of which are optional.


##Task Name


The task name is just a user-provided name for the labeling task and is only used to construct display names for various labeling artifacts (i.e. column names in the output labeled csv/dataframe)


```json title="Example"
"task_name": "CompanyEntityMatch"
```


##Task Type


The task type determines how the Autolabel library should construct the request to the LLM as well as how the LLM response should be parsed and which metrics should be computed. Currently, the library supports the following task types:


- entity_matching
- classification
- named_entity_recognition
- question_answering


```json title="Example"
"task_type": "entity_matching"
```


##Dataset


The dataset config contains information about the dataset to be labeled. Specifically, there are 4 dataset config keys:


1. label_column (optional): The label column specifies the column containing the labels for each item to use for metric computation if labels are available for the dataset
2. explanation_column (optional): The explanation column specifies the column containing explanations for each item to use for chain-of-thought prompting if it is enabled in the config.
3. delimiter (optional): This key specifies the delimiter used for parsing the dataset CSV. By default, it is assumed to be a comma: ","
4. text_column (required for named entity recognition): The text column is only necessary for named entity recognition tasks and specifies the column containing the text that we intend to label and is used for determining text spans.


```json title="Example 1: Classification task"
"dataset": {
       "label_column": "label",
       "delimiter": ","
   }
```


```json title="Example 2: Chain of thought"
   "dataset": {
       "label_column": "answer",
       "explanation_column": "explanation",
       "delimiter": ","
   }
```


```json title="Example 3: Named entity recognition task"
   "dataset": {
       "label_column": "CategorizedLabels",
       "text_column": "example",
       "delimiter": ","
   }
```






##Model


The model config contains information about the LLM provider and specific model we intend to use for labeling. There are 4 model config keys:


1. provider: This key specifies the LLM provider.
2. name: The model name specifies which of the provider's models to use for generating labels.
3. params (optional): Params is a dictionary that allows the user to configure model-specific paramaters. Here is an example model params dict:
   - max_tokens: Max tokens specifies the maximum total input and output tokens for each LLM call.
   - temperature: The temperature controls how deterministic the LLM responses should be.
   - model_kwargs: The model kwargs contains the logprobs key which, when present, configures the LLM request to have the LLM return log probabilities


4. compute_confidence (optional): This boolean determines whether to compute and output confidence scores.


```json title="Example 1: Compute confidence"
"model": {
       "provider": "openai",
       "name": "gpt-3.5-turbo",
       "compute_confidence": True
   }
```


```json title="Example 2: Defining model params"
"model": {
   "provider": "openai",
   "name": "gpt-3.5-turbo",
   "params": {
       "max_tokens": 512,
       "temperature": 0.1
   }
}
```


##Prompt


The prompt config contains information about how the prompt should be constructed in the request to the LLM. There are 9 prompt config keys.


1. task_guidelines: The task guidelines should contain a description of the specific labeling task, including any nuanced details about how to correctly label each item.
2. labels (required for some tasks): The labels defines the full list of labels for the model.
3. few_shot_examples (optional): The few shot examples is either a list or path to the CSV of possible seed examples to append to the prompt.
4. few_shot_selection (optional): The few shot selection is the specific strategy to use for selecting examples to use in the prompt. Currently, there are 3 example selection strategies implemented:

    - fixed
    - semantic_similarity
    - max_marginal_relevance
    
5. few_shot_num (optional): The few shot number determines how many seed examples to select and include in the prompt
6. example_template: The example template determines how each example should be formatted in the prompt. You can reference columns from the dataset by wrapping the column name with curly braces
7. output_guidelines (optional): The output guidelines specify how the output should be returned by the LLM (i.e. just return the label vs. format as CSV). It is not recommended to add output guidelines for most use cases as default guidelines are already set.
8. output_format (optional): The format of the output is either "csv" or "json", but it is not recommended to override the default selection.
9. chain_of_thought (optional): This boolean determines whether to use chain of thought in the prompt or not.


```json title="Example 1: Classification task"
"prompt": {
       "task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
       "labels": [
           "toxic",
           "not toxic"
       ],
       "example_template": "Input: {example}\nOutput: {label}"
   }
```


```json title="Example 2: Use seed examples"
   "prompt": {
       "task_guidelines": "You are provided with descriptions of companies from their websites, and wikipedia pages. Your job is to categorize whether the descriptions are about the same company (duplicate) or different companies (not duplicate). Your answer must be from one of the following options:\n{labels}",
       "labels": [
           "not duplicate",
           "duplicate"
       ],
       "example_template": "Company 1 description: {entity1}\nCompany 2 description: {entity2}\nDuplicate or not: {label}",
       "few_shot_examples": [
           {
               "entity1": "lac wisconsin branding 95 1 & 96 1 the rock frequency 96.1 mhz translator s 95.1 w236ag fond du lac first air date 1965 as wcwc fm at 95.9 format mainstream rock erp 4 000 watts haat 123 meters 404 ft class a facility id 54510 transmitter coordinates 43 49 10.00 n 88 43 20.00 w 43.8194444 n 88.7222222 w 43.8194444 ; 88.7222222 coordinates 43 49 10.00 n 88 43 20.00 w 43.8194444 n 88.7222222 w 43.8194444 ; 88.7222222 former callsigns wcwc fm 1965 1980 wyur 1980 1994 former frequencies 95.9 mhz 1965 affiliations cbs radio network westwood one premiere radio networks owner radio plus inc. sister stations wfdl wfdl fm wmdc webcast listen live website 961tcx . com studios in fond du lac wtcx 96.1 fm 95 1 & 96 1 the rock is a radio station broadcasting a mainstream rock music format . 1 licensed to ripon wisconsin usa the station is currently owned by radio plus inc. and features programing from cbs radio network dial global and premiere radio networks . 2 wtcx was originally on 95.9 mhz . be",
               "entity2": "closings contact next racing rocks local news breaking wiaa releases football playoffs matchups and brackets october 15 2016 local news here are the full brackets for the state of wisconsin division 1 2 seed fond du lac hosts 7 seed milwaukee washington friday october 21 at 7pm division 5 3 seed wla hosts 6 seed ... read more 10 15 16 fdl man injured in hit and run car vs. bike crash october 15 2016 local news a fond du lac man received non life threatening injuries in a car versus bicycle hit and run crash in dodge county . the dodge county sheriff s office says shortly after 8pm friday a car ... read more 10 15 16 ripon woman remains in critical condition following one vehicle crash october 15 2016 local news a ripon woman injured in a one vehicle crash after apparently falling asleep at the wheel remains in critical condition . the fond du lac county sheriff s office says 29 year old raquel amador ... read more wiaa releases football groupings october 15 2016 local news 2016 wiaa fo",
               "label": "duplicate"
           },
           {
               "entity1": "stacy spikes hamet watt headquarters new york city united states website http www.moviepass.com moviepass is a subscription based service for going to movie theaters available in the united states . the service gives members across the country the ability to see up to one 2d movie every 24 hours for a fixed monthly fee . members may choose which theaters they wish to attend and there are no blackout dates . moviepass works in nearly all movie theaters that accept the mastercard credit card making it one of the largest subscription based theater networks in america . prices vary by local market and start at 30 per month . moviepass was launched in february 2011 and is headquartered in new york city . 1 contents 1 service 2 purchasing a ticket 3 history 4 media coverage 5 references service edit the moviepass service works via a smartphone app iphone android and a specially designed reloadable debit card which is mailed to new subscribers when they sign up . purchasing a ticket edit in o",
               "entity2": "repair buy warranty get service buy warranty home warranty pricing & plans planning on moving home matters blog what s covered service professionals customer reviews benefits faqs appliance discount contract policies decor cost savers lawn & garden lifestyle quick tips real estate repair & maintenance tech close home warranty learn more what s covered service professionals faqs pricing and plans get a quote see plans planning on moving real estate plans buying a home selling a home home matters blog decor cost savers lawn & garden lifestyle quick tips real estate repair & maintenance tech our partner sites real estate professionals contractors 888 429 8247 email us log in back to top get a personalized quote explore plans in your area get covered in 3 easy steps . please correct highlighted fields request service log in create account oven on the fritz appliance breakdowns happen . get covered . get a personalized quote explore plans in your area get covered in 3 easy steps . please co",
               "label": "not duplicate"
           },
           {
               "entity1": "of over 110 gyms worldwide including 86 franchise locations in ma pa ny nj ct wa or ca tx fl ky va puerto rico and australia and is rapidly expanding across the u.s. and around the globe . contents 1 history 2 description 3 references 4 external links history edit crunch was founded in a basement level aerobics studio in new york city s east village in 1989 by doug levine . 1 with the collaboration of fitness instructors the group fitness programming was started at crunch . offerings such as hip hop aerobics co ed action wrestling and cyked yoga cycling were introduced . 2 in clubs members have access to innovative group fitness classes state of the art equipment personal and group training full service locker rooms and much more . select locations offer an exclusive crunch retail line that can also be purchased from the crunch online store . 3 in january 2014 crunch released its online workout extension called crunch live . this subscription based online video library has over 95 work",
               "entity2": "gallery esp en best rate guarantee check availability call us room only 1 800 990 8250 hotel air 1 800 219 2727 canada 1 855 478 2811 airport transportation travel agents close best rate guaranteebook your all inclusive stay hotel hotel air arrive departure adults 1 2 3 4 5 6 7 8 children 0 1 2 3 4 5 6 7 8 select property pacifica golf & spa resort the towers at pacifica sunset beach golf & spa resort ros resort & spa los cabos montecristo estates mazatl n emerald bay resort & spa emerald estates luxury villas departure country argentina australia austria bahamas belgium brazil canada chile colombia costa rica denmark ecuador finland france germany greece honduras iceland israel italy japan luxembourg mexico netherlands new zealand nicaragua norway panama paraguay peru portugal puerto rico republic of ireland republic of korea south africa spain sweden switzerland turks and caicos islands united kingdom united states uruguay venezuela departure city akron canton ohio reg . albany ny al",
               "label": "not duplicate"
           }
       ],
       "few_shot_selection": "fixed",
       "few_shot_num": 3
   }
```






##Full Example Configs
```json title="Example 1: Company Entity Match"
{
   "task_name": "CompanyEntityMatch",
   "task_type": "entity_matching",
   "dataset": {
       "label_column": "label",
       "delimiter": ","
   },
   "model": {
       "provider": "openai",
       "name": "gpt-3.5-turbo"
   },
   "prompt": {
       "task_guidelines": "You are provided with descriptions of companies from their websites, and wikipedia pages. Your job is to categorize whether the descriptions are about the same company (duplicate) or different companies (not duplicate). Your answer must be from one of the following options:\n{labels}",
       "labels": [
           "not duplicate",
           "duplicate"
       ],
       "example_template": "Company 1 description: {entity1}\nCompany 2 description: {entity2}\nDuplicate or not: {label}",
       "few_shot_examples": [
           {
               "entity1": "lac wisconsin branding 95 1 & 96 1 the rock frequency 96.1 mhz translator s 95.1 w236ag fond du lac first air date 1965 as wcwc fm at 95.9 format mainstream rock erp 4 000 watts haat 123 meters 404 ft class a facility id 54510 transmitter coordinates 43 49 10.00 n 88 43 20.00 w 43.8194444 n 88.7222222 w 43.8194444 ; 88.7222222 coordinates 43 49 10.00 n 88 43 20.00 w 43.8194444 n 88.7222222 w 43.8194444 ; 88.7222222 former callsigns wcwc fm 1965 1980 wyur 1980 1994 former frequencies 95.9 mhz 1965 affiliations cbs radio network westwood one premiere radio networks owner radio plus inc. sister stations wfdl wfdl fm wmdc webcast listen live website 961tcx . com studios in fond du lac wtcx 96.1 fm 95 1 & 96 1 the rock is a radio station broadcasting a mainstream rock music format . 1 licensed to ripon wisconsin usa the station is currently owned by radio plus inc. and features programing from cbs radio network dial global and premiere radio networks . 2 wtcx was originally on 95.9 mhz . be",
               "entity2": "closings contact next racing rocks local news breaking wiaa releases football playoffs matchups and brackets october 15 2016 local news here are the full brackets for the state of wisconsin division 1 2 seed fond du lac hosts 7 seed milwaukee washington friday october 21 at 7pm division 5 3 seed wla hosts 6 seed ... read more 10 15 16 fdl man injured in hit and run car vs. bike crash october 15 2016 local news a fond du lac man received non life threatening injuries in a car versus bicycle hit and run crash in dodge county . the dodge county sheriff s office says shortly after 8pm friday a car ... read more 10 15 16 ripon woman remains in critical condition following one vehicle crash october 15 2016 local news a ripon woman injured in a one vehicle crash after apparently falling asleep at the wheel remains in critical condition . the fond du lac county sheriff s office says 29 year old raquel amador ... read more wiaa releases football groupings october 15 2016 local news 2016 wiaa fo",
               "label": "duplicate"
           },
           {
               "entity1": "stacy spikes hamet watt headquarters new york city united states website http www.moviepass.com moviepass is a subscription based service for going to movie theaters available in the united states . the service gives members across the country the ability to see up to one 2d movie every 24 hours for a fixed monthly fee . members may choose which theaters they wish to attend and there are no blackout dates . moviepass works in nearly all movie theaters that accept the mastercard credit card making it one of the largest subscription based theater networks in america . prices vary by local market and start at 30 per month . moviepass was launched in february 2011 and is headquartered in new york city . 1 contents 1 service 2 purchasing a ticket 3 history 4 media coverage 5 references service edit the moviepass service works via a smartphone app iphone android and a specially designed reloadable debit card which is mailed to new subscribers when they sign up . purchasing a ticket edit in o",
               "entity2": "repair buy warranty get service buy warranty home warranty pricing & plans planning on moving home matters blog what s covered service professionals customer reviews benefits faqs appliance discount contract policies decor cost savers lawn & garden lifestyle quick tips real estate repair & maintenance tech close home warranty learn more what s covered service professionals faqs pricing and plans get a quote see plans planning on moving real estate plans buying a home selling a home home matters blog decor cost savers lawn & garden lifestyle quick tips real estate repair & maintenance tech our partner sites real estate professionals contractors 888 429 8247 email us log in back to top get a personalized quote explore plans in your area get covered in 3 easy steps . please correct highlighted fields request service log in create account oven on the fritz appliance breakdowns happen . get covered . get a personalized quote explore plans in your area get covered in 3 easy steps . please co",
               "label": "not duplicate"
           },
           {
               "entity1": "of over 110 gyms worldwide including 86 franchise locations in ma pa ny nj ct wa or ca tx fl ky va puerto rico and australia and is rapidly expanding across the u.s. and around the globe . contents 1 history 2 description 3 references 4 external links history edit crunch was founded in a basement level aerobics studio in new york city s east village in 1989 by doug levine . 1 with the collaboration of fitness instructors the group fitness programming was started at crunch . offerings such as hip hop aerobics co ed action wrestling and cyked yoga cycling were introduced . 2 in clubs members have access to innovative group fitness classes state of the art equipment personal and group training full service locker rooms and much more . select locations offer an exclusive crunch retail line that can also be purchased from the crunch online store . 3 in january 2014 crunch released its online workout extension called crunch live . this subscription based online video library has over 95 work",
               "entity2": "gallery esp en best rate guarantee check availability call us room only 1 800 990 8250 hotel air 1 800 219 2727 canada 1 855 478 2811 airport transportation travel agents close best rate guaranteebook your all inclusive stay hotel hotel air arrive departure adults 1 2 3 4 5 6 7 8 children 0 1 2 3 4 5 6 7 8 select property pacifica golf & spa resort the towers at pacifica sunset beach golf & spa resort ros resort & spa los cabos montecristo estates mazatl n emerald bay resort & spa emerald estates luxury villas departure country argentina australia austria bahamas belgium brazil canada chile colombia costa rica denmark ecuador finland france germany greece honduras iceland israel italy japan luxembourg mexico netherlands new zealand nicaragua norway panama paraguay peru portugal puerto rico republic of ireland republic of korea south africa spain sweden switzerland turks and caicos islands united kingdom united states uruguay venezuela departure city akron canton ohio reg . albany ny al",
               "label": "not duplicate"
           }
       ],
       "few_shot_selection": "fixed",
       "few_shot_num": 3
   }
}
```


```json title="Example 2: Banking Complaints Classification"
{
   "task_name": "BankingComplaintsClassification",
   "task_type": "classification",
   "dataset": {
       "label_column": "label",
       "delimiter": ","
   },
   "model": {
       "provider": "openai",
       "name": "gpt-3.5-turbo"
   },
   "prompt": {
       "task_guidelines": "You are an expert at understanding bank customers support complaints and queries.\nYour job is to correctly classify the provided input example into one of the following categories.\nCategories:\n{labels}",
       "output_guidelines": "You will answer with just the the correct output label and nothing else.",
       "labels": [
           "activate_my_card",
           "age_limit",
           "apple_pay_or_google_pay",
           "atm_support",
           "automatic_top_up",
           "balance_not_updated_after_bank_transfer",
           "balance_not_updated_after_cheque_or_cash_deposit",
           "beneficiary_not_allowed",
           "cancel_transfer",
           "card_about_to_expire",
           "card_acceptance",
           "card_arrival",
           "card_delivery_estimate",
           "card_linking",
           "card_not_working",
           "card_payment_fee_charged",
           "card_payment_not_recognised",
           "card_payment_wrong_exchange_rate",
           "card_swallowed",
           "cash_withdrawal_charge",
           "cash_withdrawal_not_recognised",
           "change_pin",
           "compromised_card",
           "contactless_not_working",
           "country_support",
           "declined_card_payment",
           "declined_cash_withdrawal",
           "declined_transfer",
           "direct_debit_payment_not_recognised",
           "disposable_card_limits",
           "edit_personal_details",
           "exchange_charge",
           "exchange_rate",
           "exchange_via_app",
           "extra_charge_on_statement",
           "failed_transfer",
           "fiat_currency_support",
           "get_disposable_virtual_card",
           "get_physical_card",
           "getting_spare_card",
           "getting_virtual_card",
           "lost_or_stolen_card",
           "lost_or_stolen_phone",
           "order_physical_card",
           "passcode_forgotten",
           "pending_card_payment",
           "pending_cash_withdrawal",
           "pending_top_up",
           "pending_transfer",
           "pin_blocked",
           "receiving_money",
           "Refund_not_showing_up",
           "request_refund",
           "reverted_card_payment?",
           "supported_cards_and_currencies",
           "terminate_account",
           "top_up_by_bank_transfer_charge",
           "top_up_by_card_charge",
           "top_up_by_cash_or_cheque",
           "top_up_failed",
           "top_up_limits",
           "top_up_reverted",
           "topping_up_by_card",
           "transaction_charged_twice",
           "transfer_fee_charged",
           "transfer_into_account",
           "transfer_not_received_by_recipient",
           "transfer_timing",
           "unable_to_verify_identity",
           "verify_my_identity",
           "verify_source_of_funds",
           "verify_top_up",
           "virtual_card_not_working",
           "visa_or_mastercard",
           "why_verify_identity",
           "wrong_amount_of_cash_received",
           "wrong_exchange_rate_for_cash_withdrawal"
       ],
       "few_shot_examples": "seed.csv",
       "few_shot_selection": "semantic_similarity",
       "few_shot_num": 10,
       "example_template": "Input: {example}\nOutput: {label}"
   }
}
```

