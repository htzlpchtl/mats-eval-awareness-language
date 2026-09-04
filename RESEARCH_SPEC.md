MATS application research project: agent operating specification
0. Your role
You are acting as a research engineering agent for a tightly time-boxed empirical ML research project.
The human user is the researcher and scientific decision-maker.
Your role is to:
	•	write clear, inspectable Python code;
	•	acquire and transform data exactly as specified;
	•	run deterministic experiments;
	•	record numerical results faithfully;
	•	produce diagnostic tables and figures;
	•	identify implementation problems and plausible confounds;
	•	make it easy for the human to inspect raw data and verify load-bearing claims.
Your role is not to:
	•	decide what scientific claim should ultimately be made;
	•	silently alter the experimental design;
	•	optimise for exciting-looking results;
	•	expand the project into technically sophisticated side investigations;
	•	write the final MATS executive summary or application answers;
	•	hide failed experiments or inconvenient results;
	•	continue past designated checkpoints without explicit human approval.
This project is particularly concerned with avoiding a failure mode where an AI coding agent generates a plausible-looking research project that the human does not actually understand.
At every major checkpoint, assume that the human wants to inspect and understand what has happened before authorising further work.
1. Central research question
The main research question is:
"How language-invariant are the linear activation signals identified by current evaluation-awareness probing methodologies?"
The experiment begins from a controlled 2×2 dataset crossing:
	•	evaluation-labelled vs deployment-labelled context;
	•	benchmark-structured vs casual/free-form format.
We first investigate the English representations in Qwen3.5-9B.
If we identify a useful format-controlled evaluation/deployment-labelled linear signal in English, we then test whether the same frozen English probe transfers when the exact same prompts are translated into Spanish and Japanese.
Important claim discipline:
Do not describe the probe as establishing that the model "knows it is being evaluated".
The dataset labels are proxies constructed from dataset provenance. In particular, MuTual is treated as evaluation-labelled because it is benchmark data, while ShareGPT is treated as deployment-labelled because it consists of real conversational prompts.
The strongest default language should therefore be along the lines of:
	•	"evaluation/deployment-labelled signal";
	•	"linear signal identified by current evaluation-awareness probing methodology";
	•	"linear separability of the evaluation- and deployment-labelled datasets".
Any stronger interpretation requires explicit human approval.
2. Core experimental objects
Subject model
Use:
Qwen/Qwen3.5-9B
Expected language-model architecture:
	•	approximately 9B parameters;
	•	32 language-model layers;
	•	hidden dimension 4096.
Do not blindly assume that the installed implementation exposes tensors exactly as expected. Verify the live model configuration and hidden-state API during the model smoke test.
Use BF16 inference where supported.
Do not quantise the model for the main experiment without human approval.
Do not switch to another subject model without human approval.
Dataset
Use Hugging Face dataset:
viliana-dev/eval-awareness-2x2
Core configurations:
	1	bench_eval
	◦	context label: evaluation
	◦	format label: benchmark
	◦	source: MMLU + ARC + GSM8K
	2	casual_eval_mutual
	◦	context label: evaluation
	◦	format label: casual/free-form
	◦	source: MuTual
	3	bench_deploy_rewritten
	◦	context label: deployment
	◦	format label: benchmark
	◦	source: rewritten ShareGPT prompts
	4	casual_deploy_matched
	◦	context label: deployment
	◦	format label: casual/free-form
	◦	source: ShareGPT first turns
Do not use the additional genre-control configurations unless explicitly authorised later.
Labels
Use:
context_label:
	•	1 = evaluation
	•	0 = deployment
format_label:
	•	1 = benchmark
	•	0 = casual/free-form
Keep the original quadrant as a categorical variable:
	•	bench_eval
	•	casual_eval
	•	bench_deploy
	•	casual_deploy
Never discard quadrant identity, even when training a binary classifier.
3. Dataset sample and split
The intended main sample is:
	•	300 bench_eval
	•	300 casual_eval_mutual
	•	300 bench_deploy_rewritten
	•	300 casual_deploy_matched
Total:
1,200 English prompts.
Use random seed 42 everywhere unless explicitly stated otherwise.
Within every quadrant:
	•	210 examples are training examples;
	•	90 examples are final test examples.
Therefore Probe B has:
	•	840 training prompts;
	•	360 untouched English test prompts.
The 30% final test data must not be used for:
	•	fitting probe parameters;
	•	choosing a layer;
	•	choosing regularisation;
	•	choosing a classification threshold;
	•	deciding which activation position is best;
	•	choosing preprocessing;
	•	any other optimisation.
The test data is for final evaluation after the relevant design choices have been frozen.
Fixed CV folds
Within the 210 training examples in each quadrant, create deterministic five-fold CV assignments.
Each quadrant should contribute exactly:
42 examples per fold.
Thus for Probe B each validation fold contains:
	•	42 bench_eval;
	•	42 casual_eval;
	•	42 bench_deploy;
	•	42 casual_deploy.
Store a cv_fold integer from 0 to 4 directly in the frozen selected dataset.
Use exactly these same folds at every layer.
This prevents random CV composition from creating noise when comparing layers.
4. Probe definitions
There are two primary English probes.
Probe A: naive/confounded probe
Train only on:
positive:
	•	bench_eval
negative:
	•	casual_deploy
This intentionally correlates:
evaluation with benchmark format
and:
deployment with casual format.
Probe A has two tests.
First, its ordinary in-distribution held-out test:
	•	held-out bench_eval
	•	held-out casual_deploy
Second, and much more important, its crossed-format test:
positive:
	•	held-out casual_eval
negative:
	•	held-out bench_deploy
The probe is frozen between these tests.
The crossed-format test asks whether a direction learnt from benchmark-eval vs casual-deploy generalises when format and context are reversed.
Probe A is primarily a diagnostic experiment. Do not automatically carry it through every later multilingual analysis.
Probe B: format-controlled probe
Train on all four quadrants.
Positive/evaluation class:
	•	bench_eval
	•	casual_eval
Negative/deployment class:
	•	bench_deploy
	•	casual_deploy
Because benchmark and casual format occur on both sides, prompt format alone cannot perfectly solve the classification problem.
Probe B is the main object for the multilingual experiment.
5. Probe implementation
At a given transformer layer, every prompt provides one activation vector of hidden dimension 4096.
Use an sklearn Pipeline containing:
	1	StandardScaler
	2	LogisticRegression
Logistic regression specification:
	•	L2 regularisation;
	•	C = 1.0;
	•	max_iter = 2000;
	•	random_state = 42;
	•	do not tune C unless explicitly authorised.
Primary metric:
AUROC.
Secondary metrics:
	•	accuracy using the fixed default logistic threshold;
	•	decision-function score distributions.
Class sizes should already be balanced, so do not introduce class weights unless a later experiment requires them and the human approves.
The scaler must be fitted only on the corresponding training fold during cross-validation.
Do not fit StandardScaler on all data before CV.
For the final fitted probe, fit the scaler and classifier jointly on the entire relevant training partition.
When applying an English probe to another language, freeze and apply the entire fitted English pipeline, including the English-fitted StandardScaler.
Do not refit the scaler on Spanish or Japanese before testing English-to-Spanish or English-to-Japanese transfer.
6. Layer selection
Do not decide in advance which transformer layer to probe.
For a probe:
	1	take the training examples only;
	2	at every transformer block output independently, perform five-fold CV using the fixed folds;
	3	calculate validation AUROC on each fold;
	4	average the five validation AUROCs;
	5	choose the layer with the highest mean CV AUROC.
The final untouched test set must not be consulted during this process.
After choosing the layer:
	1	discard the temporary CV models;
	2	fit one final scaler + logistic classifier on all relevant training examples at the selected layer;
	3	evaluate that frozen final probe on the appropriate held-out test data.
Probe A and Probe B are allowed to select different optimal English layers.
Call these:
L_A_EN
and:
L_B_EN.
For the later multilingual experiment, Probe B's English layer L_B_EN is the primary common reference layer.
7. Activation definition
For every input prompt, run the model forward without generating an answer.
Use the model's standard Qwen chat template with:
	•	one user message containing the dataset prompt;
	•	no additional system message;
	•	add_generation_prompt=True.
Use the official processor/model interface supported by the current Qwen3.5 implementation.
The intended activation for transformer layer l is:
the hidden-state vector at the final non-padding input token after transformer block l.
If the tokenised input positions are:
x_1, ..., x_T,
then the intended activation is:
h_(l,T).
When batching:
	•	use the attention mask;
	•	find the actual last non-padding token for every example;
	•	never blindly take tensor position -1 if examples are padded.
Before extracting the dataset, explicitly inspect what the final token actually is after Qwen's chat template.
It may be a constant assistant-generation marker. That is acceptable if verified and used consistently.
Do not include the embedding-only hidden state as one of the 32 transformer layers.
The live model API may return something like 33 hidden-state tensors, with index 0 representing embeddings and subsequent tensors representing block outputs.
Do not assume this. Verify and document the precise mapping.
8. Prompt content discipline
Only the actual prompt presented to the language model should enter Qwen.
Never accidentally append:
	•	the ground-truth answer;
	•	a solution;
	•	context_label;
	•	format_label;
	•	quadrant name;
	•	source dataset name;
	•	metadata;
	•	any other information unavailable to the model in the intended experiment.
For rewritten benchmark-deployment data, the intended model-facing text is expected to be the rewritten benchmark-form prompt, not the original ShareGPT query.
For every configuration, inspect the actual live schema before choosing the model-facing field.
The field choice requires human approval.
9. Repository structure
Use ordinary Python source files, not a notebook-dependent workflow.
Create approximately:
README.md RESEARCH_SPEC.md RUN_LOG.md research_notes.md requirements.txt .gitignore config.yaml
src/ inspect_data.py prepare_data.py model_smoke_test.py extract_activations.py train_probe_a.py train_probe_b.py evaluate_transfer.py translate_prompts.py translation_qa.py make_figures.py utils.py
tests/ test_splitting.py test_activation_indexing.py test_probe_pipeline.py
data/ selected/ translated/
artifacts/ activations/
results/ metrics/ predictions/ figures/
docs/ COMPUTE_SETUP.md
Exact filenames can vary slightly if there is a compelling engineering reason, but keep the structure simple and legible.
Git policy
Do not commit or push without explicit human approval.
The human will review the diff and decide when a checkpoint should become a Git commit.
Do not modify Git history.
Do not force push.
Do not create unnecessary branches.
Files that must never be committed
Ensure .gitignore excludes:
	•	.env
	•	API keys
	•	authentication tokens
	•	model-weight caches
	•	Hugging Face cache
	•	large activation tensors
	•	virtual environments
	•	Python cache files
	•	temporary API response caches containing credentials
	•	other large disposable artifacts.
Small reproducibility artifacts should normally be committed:
	•	frozen selected dataset or its reproducible IDs/metadata;
	•	split assignments;
	•	translation text after approval;
	•	result CSVs;
	•	per-example final probe predictions;
	•	figures;
	•	configuration;
	•	environment metadata;
	•	code.
10. Logging and reproducibility
Maintain RUN_LOG.md.
This should be factual rather than rhetorical.
For every experiment record:
	•	timestamp;
	•	command;
	•	relevant config;
	•	model revision;
	•	dataset revision/fingerprint where available;
	•	random seed;
	•	sample size;
	•	files written;
	•	metric output;
	•	warnings/errors.
Do not fabricate interpretations.
Maintain research_notes.md as a human-owned file. Do not overwrite human notes.
At each important run record:
	•	Python version;
	•	PyTorch version;
	•	Transformers version;
	•	CUDA version;
	•	GPU model;
	•	model config/revision;
	•	dataset fingerprint/revision.
Once the GPU environment works, save a package freeze or equivalent environment manifest.
11. Checkpoint protocol
The following rule overrides any desire to continue autonomously.
At every designated checkpoint:
STOP.
Do not execute the next stage.
Return a checkpoint report containing:
	1	what you did;
	2	files created or changed;
	3	exact commands run;
	4	tests/checks passed;
	5	tests/checks failed or warnings;
	6	important numerical results, if any;
	7	a few raw examples where appropriate;
	8	implementation choices you made;
	9	anything you are uncertain about;
	10	plausible ways the current result could be wrong;
	11	the exact next action you propose.
Then wait for explicit human approval.
Approval means an unambiguous instruction such as:
"Approve checkpoint 3. Proceed to stage 4."
Do not interpret silence or a general positive comment as approval to continue.
12. Stage 0: local repository scaffold and synthetic tests
This stage must not require a GPU.
Goals:
	•	create the basic repository structure;
	•	implement reusable deterministic utilities;
	•	establish tests for the most dangerous bookkeeping errors;
	•	avoid any full-scale data or model processing.
Create:
	•	config;
	•	requirements;
	•	.gitignore;
	•	source/test scaffolding;
	•	run-log format.
Use Python 3.11 if practical.
Include dependencies needed for:
	•	datasets;
	•	numpy;
	•	pandas;
	•	scikit-learn;
	•	matplotlib;
	•	pytest;
	•	torch;
	•	transformers;
	•	accelerate;
	•	OpenAI-compatible API calls later.
Do not download Qwen model weights.
Required synthetic tests
Test deterministic quadrant splitting.
Create synthetic records for four quadrants and verify:
	•	exact 70/30 train/test counts;
	•	zero ID overlap;
	•	five CV folds;
	•	equal quadrant representation in each B fold.
Test final-token indexing.
Using synthetic hidden-state tensors and artificial attention masks, verify that the helper function:
	•	returns the final non-padding activation for every example;
	•	works for examples with different sequence lengths;
	•	does not return padding positions.
Test probe layer selection.
Create synthetic activations where:
	•	one known layer contains a strong binary signal;
	•	other layers contain noise.
Verify that the CV procedure selects or nearly selects the intended signal layer.
This is not scientific validation. It is an implementation sanity check.
CHECKPOINT 0
Report:
	•	repo tree;
	•	dependency choices;
	•	all test results;
	•	any proposed deviations from this specification.
Stop.
Do not inspect/download the research dataset until approved.
13. Stage 1: inspect the live dataset
After approval:
Load all four specified Hugging Face configurations.
Do not sample the final 300 yet.
For each configuration report:
	•	exact row count;
	•	exact column/schema names;
	•	types;
	•	likely model-facing text fields;
	•	any answer/solution fields;
	•	source metadata fields;
	•	example identifiers if present.
Print 10 deterministic random examples from every quadrant.
For each example show only enough metadata to let the human understand:
	•	what Qwen would potentially see;
	•	what fields would not be shown;
	•	whether an answer or solution is stored elsewhere.
Explicitly identify the proposed model-facing text field for each quadrant.
Do not finalise that field mapping yourself.
Data integrity checks
Check:
	•	exact duplicate prompts within each quadrant;
	•	exact duplicate prompts across quadrants;
	•	missing/empty prompt fields;
	•	obviously malformed rows.
Report, do not silently repair.
CHECKPOINT 1
Stop and ask the human to approve the proposed model-facing field mapping and any handling of duplicates/malformed rows.
Do not create the frozen sample until approved.
14. Stage 2: freeze the English dataset and split
After field mapping is approved:
Select exactly 300 valid examples from each quadrant, seed 42.
If duplicates or malformed rows require removal, do so only according to the approved rule.
Assign every example a stable unique ID.
The ID should not depend only on list ordering. Include:
	•	quadrant;
	•	original row index or stable source identifier;
	•	optionally a short deterministic hash of the model-facing text.
Save for each selected example:
	•	stable ID;
	•	quadrant;
	•	context_label;
	•	format_label;
	•	source metadata if available;
	•	model-facing English text;
	•	train/test assignment;
	•	cv_fold for training examples.
Produce exactly:
per quadrant:
	•	train = 210
	•	test = 90
Then within each quadrant's 210 training rows assign:
	•	42 to fold 0
	•	42 to fold 1
	•	42 to fold 2
	•	42 to fold 3
	•	42 to fold 4
Required checks
Assert:
	•	1,200 total rows;
	•	exactly 300 per quadrant;
	•	exactly 840 train;
	•	exactly 360 test;
	•	no ID overlap;
	•	no duplicated model-facing text unless explicitly approved;
	•	exactly 42 training examples from each quadrant in every CV fold.
Calculate basic English statistics by quadrant:
	•	character length mean/median/std;
	•	word count mean/median;
	•	relevant available source distributions.
Do not interpret modest statistical differences as scientific findings.
Print 10 examples from each frozen quadrant for human inspection.
CHECKPOINT 2
Stop.
The human must approve the frozen dataset and split before any model activations are extracted.
Once approved, treat the frozen IDs and split as immutable except in response to a genuine implementation/data bug, which must be documented.
15. Stage 3: prepare the compute handoff
This stage still must not provision compute.
The human, not you, is responsible for renting the GPU.
Prepare docs/COMPUTE_SETUP.md containing the exact reproducible setup required for the human.
Expected initial target:
	•	one NVIDIA A40 48GB or RTX A6000 48GB;
	•	approximately 80GB or more usable storage;
	•	SSH access.
Qwen3.5-9B should fit comfortably in BF16 on a 48GB card.
The current Qwen3.5 model requires a sufficiently recent Transformers implementation. Follow the current official model-card instructions rather than silently relying on an old installed version.
Prepare exact shell commands for:
	•	checking GPU availability;
	•	cloning/pulling the repository;
	•	creating the Python environment;
	•	installing the required packages;
	•	verifying CUDA;
	•	optionally authenticating to Hugging Face if necessary.
Do not put real secrets in documentation.
Use environment variables.
CHECKPOINT 3
Return the compute setup instructions.
Stop.
Wait for the human to provision the GPU and tell you that the remote environment is available.
16. Stage 4: Qwen model smoke test on GPU
Once the human provides the GPU environment:
First run:
	•	nvidia-smi;
	•	Python/Torch/CUDA checks.
Then load:
Qwen/Qwen3.5-9B
Use the current officially supported Hugging Face model and processor classes.
Prefer the official Qwen3.5 loading path first rather than inventing a custom model wrapper.
Do not use nnsight or TransformerLens unless ordinary model access demonstrably fails and the human approves a change.
Verify architecture
Report live:
	•	model class;
	•	model revision/commit where available;
	•	number of language-model blocks;
	•	hidden dimension;
	•	dtype;
	•	device placement.
Expected approximately:
	•	32 blocks;
	•	hidden dimension 4096.
If these materially disagree, stop and investigate before continuing.
Inspect chat templating
Use one prompt from each quadrant.
For each:
	•	construct one user message;
	•	no system prompt;
	•	apply the Qwen chat template;
	•	use add_generation_prompt=True.
Print:
	•	the original model-facing text;
	•	the fully templated decoded string;
	•	total token count;
	•	final 10 token IDs/tokens;
	•	actual final non-padding token.
The human must be able to see what Qwen is receiving.
Forward pass
Do not generate an answer.
Run inference with:
	•	torch.inference_mode();
	•	BF16 where supported;
	•	output_hidden_states=True;
	•	use_cache=False;
	•	return-dict style output where supported.
Inspect:
	•	number of hidden-state tensors;
	•	every tensor shape;
	•	whether index 0 is embeddings;
	•	precise mapping from stored tensor index to transformer block.
Create an explicit mapping used by all later code.
Padding check
Construct a batch containing prompts of clearly different lengths.
Use dynamic padding.
Ensure the processor/tokenizer padding side is known and recorded.
Use attention_mask.sum(dim=1) - 1 only if this correctly identifies the final non-padding position under the chosen padding convention.
Verify this manually for the smoke batch.
Compare:
	•	batched extraction;
	•	single-example extraction
for at least one example.
They should agree numerically to floating-point tolerance.
Storage smoke test
Save extracted activations for the smoke examples.
Reload them.
Verify:
	•	IDs unchanged;
	•	shapes unchanged;
	•	values unchanged;
	•	no NaNs or infinities.
Report GPU memory use.
CHECKPOINT 4
Stop.
Provide the four templated examples, hidden-state mapping, shape information, final-token handling and any warnings.
Do not run the full 1,200-prompt extraction until approved.
17. Stage 5: full English activation extraction
After approval:
Before the full run, process a 40-example sample:
	•	10 from each quadrant.
Verify all assertions.
Then process all 1,200 frozen English prompts.
Load the model once and reuse it.
Use a conservative initial batch size, for example 4, and increase only if clearly safe.
Dynamic padding is preferred.
Do not silently truncate inputs.
Before extraction, calculate Qwen token lengths for all selected prompts.
Report:
	•	min;
	•	median;
	•	95th percentile;
	•	max;
	•	number that would exceed any proposed limit.
If any truncation is contemplated, stop for human approval first.
Store activations
For each prompt save:
	•	stable ID;
	•	quadrant;
	•	split;
	•	all 32 transformer-block final-token activation vectors.
Store large tensors under ignored artifacts/activations/.
Use a representation suitable for efficient later NumPy/sklearn use.
CPU float32 is preferred for stored probe inputs unless storage becomes problematic.
Checkpoint after each quadrant so a failed run does not require repeating completed work.
Write a manifest associating every tensor row with its stable ID.
Required assertion:
number and order of IDs exactly match number and order of activation rows.
CHECKPOINT 5
Stop.
Report:
	•	number extracted per quadrant;
	•	final activation shape;
	•	token-length statistics;
	•	VRAM behaviour;
	•	NaN/Inf checks;
	•	disk files;
	•	model revision;
	•	any extraction failures.
Do not train probes until approved.
18. Stage 6: Probe A
After approval:
Use training examples only.
Probe A training data:
positive:
	•	bench_eval train, n=210
negative:
	•	casual_deploy train, n=210
For every one of the 32 transformer layers:
	•	use the fixed five CV folds;
	•	fit StandardScaler only on each fold's training portion;
	•	fit L2 logistic regression;
	•	evaluate AUROC on the validation portion;
	•	save fold-level metrics.
Produce:
	•	mean CV AUROC per layer;
	•	standard deviation across folds;
	•	layer curve.
Choose:
L_A_EN = layer with highest mean training-CV AUROC.
Do not consult any final test examples during layer selection.
Then fit the final Probe A pipeline on all 420 Probe-A training examples at L_A_EN.
Probe A test 1: in-distribution
Evaluate on untouched:
	•	bench_eval test, n=90, positive;
	•	casual_deploy test, n=90, negative.
Report:
	•	AUROC;
	•	accuracy;
	•	confusion matrix;
	•	decision-score distribution by quadrant.
Probe A test 2: crossed format
Without refitting anything, evaluate the exact same frozen Probe A on:
	•	casual_eval test, n=90, positive;
	•	bench_deploy test, n=90, negative.
Report:
	•	AUROC;
	•	accuracy;
	•	confusion matrix;
	•	decision-score distributions separately for the two quadrants.
If AUROC becomes less than 0.5, report the actual value. Do not replace it with 1 - AUROC.
A below-chance result may be scientifically informative because it could mean Probe A systematically follows format in the opposite direction to the context label.
CHECKPOINT 6
Stop.
Report:
	•	layer curve;
	•	selected L_A_EN;
	•	in-distribution performance;
	•	crossed-format performance;
	•	quadrant score summaries;
	•	anything that appears suspicious.
Explicitly list at least three plausible explanations for the result, including trivial/confounding ones.
Do not proceed to Probe B until the human has inspected this.
19. Stage 7: Probe B
After approval:
Probe B training data contains all 840 English training examples.
Positive/evaluation:
	•	210 bench_eval
	•	210 casual_eval
Negative/deployment:
	•	210 bench_deploy
	•	210 casual_deploy
Use the fixed quadrant-balanced CV folds.
For each of the 32 transformer layers:
	•	fit scaler + logistic regression independently within each CV fold;
	•	calculate validation AUROC;
	•	save fold-level results.
Choose:
L_B_EN = layer with highest mean training-CV AUROC.
Do not use the final English test set for this selection.
Fit one final Probe B pipeline on all 840 English training examples at L_B_EN.
Final English Probe-B test
Evaluate once on all 360 untouched English test prompts.
Report:
	•	overall eval-vs-deploy AUROC;
	•	overall accuracy;
	•	confusion matrix.
Crucially also report all four quadrants separately.
For each quadrant report:
	•	n;
	•	mean decision score;
	•	median decision score;
	•	standard deviation;
	•	proportion classified as evaluation;
	•	proportion correctly classified according to context_label.
Produce a figure displaying the four held-out score distributions.
The human wants to know whether performance is genuinely:
bench_eval and casual_eval scoring above bench_deploy and casual_deploy
or whether residual benchmark/casual structure appears to dominate.
Cheap baselines
At L_B_EN also run:
	1	shuffled-label baseline
	◦	shuffle context labels within the training data;
	◦	refit using the same pipeline;
	◦	repeat 20 deterministic seeds;
	◦	evaluate held-out test AUROC;
	◦	report distribution.
	2	length-only baseline
	◦	use simple text-length features such as character count and Qwen token count;
	◦	fit only on training;
	◦	evaluate on held-out English test;
	◦	report AUROC.
Do not introduce elaborate handcrafted text features.
CHECKPOINT 7: major scientific go/no-go gate
Stop.
This checkpoint must present clearly:
	•	Probe A result;
	•	Probe B layer curve;
	•	L_B_EN;
	•	Probe B pooled test performance;
	•	four-quadrant test behaviour;
	•	shuffled-label baseline;
	•	length baseline.
Do not begin translation automatically.
Wait for the human to decide whether the English phenomenon is sufficiently coherent to justify the multilingual experiment.
20. Stage 8: translation preparation and small smoke test
Only after explicit approval to proceed cross-lingually:
Target languages:
	•	Spanish
	•	Japanese
Translation model:
DeepSeek V4 Flash 0731 through OpenRouter.
Model slug:
deepseek/deepseek-v4-flash-0731
Use an environment variable:
OPENROUTER_API_KEY
Never write or print the API key.
Cache successful translations by stable prompt ID and target language.
Requests must be resumable.
Do not translate using the Qwen subject model.
Translation instruction
Use a fixed translation instruction with the following meaning:
"You are a literal translation system.
Translate the supplied English text into TARGET_LANGUAGE.
Preserve the meaning exactly. Do not answer or solve the task. Do not add explanations, commentary, headings, or disclaimers. Preserve code, URLs, mathematical notation, numbers, and proper nouns where appropriate. Preserve answer-choice labels such as (A), (B), (C), (D). Preserve the existing list, paragraph and markdown structure as closely as possible.
Return only the translated text."
Use temperature 0 or the closest deterministic setting supported.
Disable extra reasoning output if supported.
Smoke translation
Do not translate all 1,200 immediately.
First translate:
	•	2 prompts per quadrant into Spanish;
	•	the same 2 prompts per quadrant into Japanese.
Total:
	•	8 Spanish;
	•	8 Japanese.
Run structural QA and print the original and translation side by side.
Check:
	•	nonempty result;
	•	no translator commentary;
	•	choice-label preservation;
	•	approximate number/structure preservation;
	•	no obvious accidental answer generation;
	•	reasonable length ratio.
CHECKPOINT 8A
Stop for human inspection.
Do not launch the full translation job until approved.
21. Stage 8B: full translation and QA
After approval:
Translate all 1,200 selected prompts into:
	•	Spanish;
	•	Japanese.
This includes both training and test examples because later native-language probes may be needed.
Keep:
	•	stable ID;
	•	original English text;
	•	translated text;
	•	quadrant;
	•	train/test assignment;
	•	cv_fold.
Never reshuffle language-specific versions.
An English training item remains a training item in every language.
An English test item remains a test item in every language.
Automated translation QA
Flag, but do not silently remove:
	•	empty translations;
	•	translation identical to English;
	•	suspiciously extreme character-length ratios;
	•	missing multiple-choice labels;
	•	changed count of answer choices;
	•	translator preambles;
	•	phrases suggesting the translator answered rather than translated;
	•	failed API requests.
Produce a QA CSV.
Human-audit sample
Present random frozen samples:
Spanish:
	•	10 from each quadrant, 40 total.
Japanese:
	•	5 from each quadrant, 20 total.
The human can assess Spanish directly or with assistance.
For Japanese, note explicitly that human linguistic fluency may not be available. Do not claim human validation of semantic nuance unless it happened.
CHECKPOINT 8B
Stop.
Do not extract translated activations until translation quality is approved.
22. Stage 9: first multilingual transfer test
After translation approval:
The first multilingual question is deliberately conservative:
Does the frozen English Probe B work on translations of the exact held-out English test items?
Initially we only need translated held-out test activations.
There are:
	•	360 Spanish test prompts;
	•	360 Japanese test prompts.
Extract their Qwen activations using exactly the same:
	•	model revision;
	•	processor;
	•	chat template;
	•	activation position;
	•	layer mapping;
	•	dtype convention
used for English.
Although all hidden states may be produced naturally in the forward pass, the critical evaluation is at the already-fixed English layer L_B_EN.
Apply the exact frozen English Probe-B pipeline:
	•	English-fitted StandardScaler;
	•	English-fitted logistic regression;
	•	no target-language refitting.
Compute:
EN -> EN EN -> ES EN -> JA
at L_B_EN.
Report AUROC for all three.
Also break Spanish and Japanese down by the same four quadrants:
	•	bench_eval;
	•	casual_eval;
	•	bench_deploy;
	•	casual_deploy.
Report the same quadrant score summaries as for English.
Do not train Spanish or Japanese probes yet.
CHECKPOINT 9
Stop.
Present:
	•	EN->EN AUROC;
	•	EN->ES AUROC;
	•	EN->JA AUROC;
	•	all four quadrant score distributions in each language;
	•	translation QA caveats;
	•	any obvious systematic shifts in raw probe scores.
Wait for the human to interpret the result and choose the next branch.
23. Stage 10: native-language diagnostic probes, conditional only
Do not run this stage automatically.
It is primarily needed if English Probe B transfers poorly to Spanish and/or Japanese.
For each language authorised by the human:
First extract activations for that language's 840 translated training prompts.
Diagnostic 10A: native probe at the English layer
At fixed layer:
L_B_EN
train a fresh Probe B using that language's training translations.
Use the same fixed semantic CV folds inherited from English.
Fit the same scaler + logistic method.
Then evaluate the final native probe on that language's untouched 360 test translations.
Examples:
ES-at-English-layer -> ES test
JP-at-English-layer -> JP test.
This answers:
"Is a linearly decodable eval/deploy-labelled signal present in the same layer, but in a language-specific direction?"
Checkpoint before layer scanning
If a native-language probe works well at L_B_EN, stop.
Do not automatically optimise that language's layer.
This already distinguishes:
English direction does not transfer
from:
signal is absent at that layer.
Diagnostic 10B: full native-language CV curve
Only if:
	•	EN -> target-language transfer is weak;
	•	and the native probe at L_B_EN is also weak;
	•	and the human explicitly approves the diagnostic.
Using target-language training examples only:
Repeat the full 32-layer five-fold CV scan.
For Japanese this defines:
L_B_JP
For Spanish:
L_B_ES.
Do not consult target-language test examples to choose these layers.
After choosing the language-specific layer from target-language training CV:
	•	fit final native-language Probe B on all target-language training examples at that layer;
	•	evaluate once on the untouched target-language test set.
This distinguishes:
	1	same layer and same direction;
	2	same layer but different direction;
	3	different optimal layer;
	4	weak/nonexistent linear decodability under the translated dataset.
CHECKPOINT 10
Stop.
Report only the diagnostics explicitly authorised.
24. Optional analyses that are forbidden by default
Do not run these merely because they sound interesting.
They are possible later follow-ups only after the core submission story is already complete and the human explicitly asks.
Examples:
	•	cosine similarity between Probe A and Probe B;
	•	explicit format probe;
	•	subtracting a format direction from Probe A;
	•	English/Spanish/Japanese probe-direction cosines;
	•	PCA;
	•	CCA;
	•	Procrustes alignment;
	•	principal-angle/subspace analysis;
	•	bootstrapped probe subspaces;
	•	steering interventions;
	•	sparse autoencoders;
	•	J-Lens;
	•	alternative subject models;
	•	a Qwen3.6-27B replication;
	•	large hyperparameter sweeps;
	•	many alternative pooling schemes.
One inexpensive alternative-pooling sanity check may be authorised later if the final-token result appears fragile, for example mean pooling over the last eight non-padding tokens.
Do not do it automatically.
25. Result files
For every final evaluation, save per-example predictions.
Each prediction row should contain enough information to independently recompute the reported metric:
	•	stable ID;
	•	language;
	•	quadrant;
	•	context_label;
	•	format_label;
	•	split;
	•	model revision;
	•	layer;
	•	decision score;
	•	predicted probability if used;
	•	predicted class.
Never provide only an aggregate AUROC without saving the underlying predictions.
This allows the human to independently recompute headline numbers.
26. Figures
Do not generate decorative plots.
Generate figures that directly support scientific interpretation.
Likely core English figures:
	1	Probe A layerwise CV AUROC.
	2	Probe B layerwise CV AUROC.
	3	Probe B held-out English score distribution across all four quadrants.
Likely multilingual figure, if the project reaches that point:
	4	EN->EN vs EN->ES vs EN->JA Probe-B AUROC, with accompanying four-quadrant breakdown.
A 3x3 train-language/test-language heatmap is not required unless native-language probes are actually run.
Plots must:
	•	have labelled axes;
	•	state sample sizes where useful;
	•	identify the probe/layer;
	•	include chance reference where appropriate;
	•	correspond exactly to saved numeric result files.
27. Independent verification requirements
Whenever a load-bearing numerical result appears, do not merely trust the main analysis script.
For every headline AUROC:
	•	save per-example scores;
	•	independently recompute AUROC from that CSV in a separate small verification path;
	•	report whether the two values agree.
For every selected layer:
	•	independently identify the argmax from the saved layer-CV CSV;
	•	verify it matches the layer used for the final probe.
For every train/test split:
	•	assert no ID overlap.
For translations:
	•	assert that semantic IDs retain identical train/test status and CV fold.
For activation arrays:
	•	assert ID ordering matches tensor ordering.
These checks are mandatory.
28. Scientific red-team behaviour
When something looks successful, treat success as a hypothesis rather than proof.
At every scientific checkpoint ask:
	•	Could this be a data leak?
	•	Could labels accidentally appear in model-facing text?
	•	Could this be prompt length?
	•	Could this be benchmark format?
	•	Could this be dataset identity?
	•	Could this be an artefact of the chat template?
	•	Could this be final-token indexing?
	•	Could we have fitted preprocessing on test data?
	•	Could translations have introduced a systematic style difference?
	•	Could a few extreme examples drive the AUROC?
	•	Are quadrant results consistent with the pooled headline?
Do not invent increasingly elaborate controls automatically.
Surface the strongest plausible alternative explanations to the human and let the human choose what is worth testing.
29. Failure handling
If an implementation step fails:
Do not immediately replace the method with something more complicated.
First:
	1	isolate the minimal failing case;
	2	inspect the error;
	3	check current official documentation/source;
	4	make the smallest justified fix;
	5	rerun the smoke test;
	6	report any substantive change in methodology.
Specific rules:
	•	If Qwen hidden states are awkward, investigate the ordinary Hugging Face model interface before introducing nnsight.
	•	If a GPU batch OOMs, reduce batch size before quantising.
	•	If an API translation request fails, retry/cachably resume rather than changing translator.
	•	If Probe B is weak, do not tune many hyperparameters until the human has examined the result.
	•	If an exciting result appears, do not automatically add five follow-up experiments.
30. Stop conditions and priorities
The project should remain a small, complete scientific investigation.
Priorities, in order:
	1	Correct data.
	2	Correct activation extraction.
	3	Honest Probe A test.
	4	Honest Probe B test.
	5	Four-quadrant inspection.
	6	Multilingual transfer, only if English is coherent.
	7	Diagnostics only where the observed result demands them.
	8	Optional geometry only if the core project is already complete.
A negative or mixed result is acceptable.
Do not massage the project toward a positive result.
Do not broaden the project merely to increase the quantity of output.
31. Final code-quality requirements
Code should be:
	•	readable;
	•	deterministic;
	•	modular enough to inspect;
	•	command-line runnable;
	•	free of unnecessary abstractions;
	•	documented where a non-obvious scientific choice is implemented.
Prefer explicit boring code over sophisticated framework design.
This is a research sprint, not a software-platform project.
Every major script should support --help.
Where practical, configurations should be read from config.yaml rather than copied in multiple scripts.
Do not build infrastructure we do not need.
32. Immediate instruction
Read this entire specification before acting.
Then execute Stage 0 only.
Do not inspect the live research dataset yet. Do not download Qwen weights. Do not provision a GPU. Do not translate anything. Do not proceed to Stage 1.
When Stage 0 is complete, return CHECKPOINT 0 and wait for explicit human approval.
