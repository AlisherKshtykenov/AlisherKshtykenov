# ROADMAP-ML

Мой личный трекер пути в ML: что уже изучено, что в работе, что впереди и где сейчас проблемы.
Этот файл собирается сам из `roadmap_tracker.csv` скриптом `scripts/update_progress.py` — руками не править.

<!-- progress:start -->
![ROADMAP-ML progress](assets/progress.png)
<!-- progress:end -->

## Прогресс

**Готовность: 1%** · изучено 0 из 265 тем · в работе 4 · в планах 261 · закреплено повторениями: 0%

| Этап | Тем | Изучено | В работе | В планах | Готовность | Повторения |
|---|---:|---:|---:|---:|---:|---:|
| eda | 10 | 0 | 0 | 10 | 0% | 0% |
| preprocessing | 65 | 0 | 0 | 65 | 0% | 0% |
| model | 91 | 0 | 1 | 90 | 1% | 0% |
| activation | 15 | 0 | 1 | 14 | 3% | 0% |
| loss | 25 | 0 | 1 | 24 | 2% | 0% |
| optimizer | 24 | 0 | 1 | 23 | 2% | 0% |
| metrics | 35 | 0 | 0 | 35 | 0% | 0% |
| **всего** | 265 | 0 | 4 | 261 | 1% | 0% |

_Готовность — среднее `learned` (изучено = 1, в работе = 0.5, в планах = 0). Повторения — `repeated` / 4, по ним красятся кубики на картинке._

## Изучено

_Пока пусто._

## В работе

_Код уже есть, но статус в шапке ещё не `ready`._

| Тема | Этап | Файл | Где применяется |
|---|---|---|---|
| Logistic Regression (логистическая регрессия) | model | [`logistic_regression.py`](src/model/supervised/classification/logistic_regression/logistic_regression.py) | — |
| Sigmoid (сигмоида) | activation | [`sigmoid.py`](src/activations/output/sigmoid.py) | — |
| Log Loss / Binary Cross-Entropy (бинарная кросс-энтропия) | loss | [`log_loss.py`](src/losses/classification/log_loss.py) | — |
| Batch GD (полный градиентный спуск) | optimizer | [`batch_gd.py`](src/optimizers/gradient_descent/batch_gd.py) | — |

## Планы

_Что ещё предстоит выучить: файл заведён, но кода пока нет (или файла нет совсем)._

<details><summary><b>eda</b> — 10</summary>

- **distributions**: Histogram (гистограмма), KDE (оценка плотности ядром), Boxplot (ящик с усами)
- **correlations**: Pearson (корреляция Пирсона), Spearman (корреляция Спирмена), Cramér's V (V Крамера), Mutual Information (взаимная информация)
- **multivariate**: Pairplot (попарные графики), Heatmap (тепловая карта), PCA projection (проекция PCA)

</details>

<details><summary><b>preprocessing</b> — 65</summary>

- **missing**: Missing report (отчёт о пропусках), Missing pattern (структура пропусков), Drop (удаление), Mean / Median / Mode (среднее / медиана / мода), Constant (константа), Forward / Backward fill (заполнение соседним), Interpolation (интерполяция), KNN Imputer (заполнение по соседям), Iterative Imputer, MICE (итеративное заполнение), Missing Indicator (индикатор пропуска), ffill_bfill
- **outliers**: IQR (межквартильный размах), Z-score (стандартизованное отклонение), Modified Z-score (модифицированный Z), Isolation Forest (изолирующий лес), LOF (локальный фактор выброса), DBSCAN (кластеризация по плотности), Remove (удаление), Clip / Winsorizing (обрезка), Log transform (логарифм), Impute (замена как пропуск), Flag (пометка)
- **encoding**: One-Hot (прямое кодирование), Label (меточное), Ordinal (порядковое), Target (по целевой переменной), Frequency (частотное), Binary (двоичное), Hashing (хеширование)
- **scaling**: StandardScaler (стандартизация), MinMaxScaler (минимакс), RobustScaler (устойчивое масштабирование), MaxAbsScaler (по максимуму модуля), Normalizer (нормализация строк)
- **transformation**: Log (логарифм), Box-Cox (преобразование Бокса–Кокса), Yeo-Johnson (преобразование Йео–Джонсона), Quantile (квантильное)
- **feature_engineering**: Polynomial (полиномиальные), Interactions (взаимодействия), Binning (разбиение на интервалы), Datetime features (признаки из даты), Aggregations (агрегаты)
- **feature_selection**: Variance Threshold (порог дисперсии), Correlation (корреляция), Chi² (хи-квадрат), Mutual Information (взаимная информация), RFE (рекурсивное исключение признаков), Forward Selection (прямой отбор), Backward Elimination (обратное исключение), Lasso (L1-регуляризация), Tree Importance (важность по дереву)
- **splitting**: Holdout (отложенная выборка), Stratified (стратифицированное), K-Fold (k-кратная кросс-валидация), Stratified K-Fold (стратифицированная k-кратная), Group K-Fold (k-кратная по группам), Time Series Split (разбиение по времени), Leave-One-Out (исключение по одному)
- **imbalance**: Random Oversampling (случайное дублирование), SMOTE (синтетическое дополнение), ADASYN (адаптивное синтетическое), Random Undersampling (случайное удаление), Tomek Links (связи Томека), Class Weights (веса классов)

</details>

<details><summary><b>model</b> — 90</summary>

- **classification**: k-NN (k ближайших соседей), Naive Bayes (наивный Байес), Decision Tree (дерево решений), Random Forest (случайный лес), Gradient Boosting (градиентный бустинг: XGBoost, LightGBM, CatBoost), SVM (метод опорных векторов), knn
- **regression**: Linear Regression (линейная регрессия), Ridge / Lasso (регрессия с регуляризацией), Polynomial Regression (полиномиальная), k-NN, Decision Tree, Random Forest, Gradient Boosting, SVR (опорные векторы для регрессии), knn, lasso, ridge
- **ranking**: LambdaMART, RankNet, бустинг с ranking loss
- **clustering**: K-Means, Hierarchical (иерархическая), DBSCAN / HDBSCAN (по плотности), Gaussian Mixture (смесь гауссиан), Mean Shift, Spectral Clustering (спектральная), dbscan, kmeans, spectral
- **dimensionality_reduction**: PCA (метод главных компонент), t-SNE, UMAP, LDA (линейный дискриминантный анализ, вообще-то он с учителем), Autoencoder (автоэнкодер), tsne
- **anomaly_detection**: Isolation Forest (изолирующий лес), One-Class SVM, LOF (локальный фактор выброса), автоэнкодер по ошибке восстановления
- **association_rules**: Apriori, FP-Growth
- **control**: Q-learning, SARSA, DQN (глубокий Q-learning), Policy Gradient, PPO, Actor-Critic, MCTS (поиск по дереву Монте-Карло)
- **deep_learning**: Image Classification (классификация изображений), Object Detection (детекция объектов), Segmentation (сегментация), Face Recognition (распознавание лиц), ASR, Speech Recognition (распознавание речи), TTS, Text-to-Speech (синтез речи), NLP: Text Classification (классификация текста), Seq2Seq: Machine Translation (машинный перевод), Generative models (генеративные модели), Time Series Forecasting (прогноз временных рядов), Recommender Systems (рекомендательные системы)
- **face_recognition**: arcface, facenet
- **generative**: diffusion, gan, vae
- **image_classification**: efficientnet, resnet, vit
- **object_detection**: detr, faster_rcnn, yolo
- **recommender**: collaborative_filtering, matrix_factorization, two_tower
- **segmentation**: deeplab, mask_rcnn, unet
- **seq2seq**: encoder_decoder, transformer
- **speech_recognition**: ctc_lstm, whisper
- **text_classification**: bert, lstm
- **text_to_speech**: fastspeech, hifigan, tacotron2, vits
- **time_series**: arima, lstm, prophet

</details>

<details><summary><b>activation</b> — 14</summary>

- **output**: Identity / Linear (тождественная), Softmax (софтмакс), Tanh (гиперболический тангенс)
- **saturating**: Sigmoid (сигмоида), Tanh (гиперболический тангенс)
- **relu_family**: ReLU (выпрямитель), Leaky ReLU (протекающий ReLU), PReLU (параметрический ReLU), ELU (экспоненциальный линейный), SELU (масштабированный ELU)
- **smooth**: GELU (гауссов линейный), Swish / SiLU (свиш), Mish (миш), Softplus (мягкий плюс)

</details>

<details><summary><b>loss</b> — 24</summary>

- **regression**: MSE, Mean Squared Error (среднеквадратичная ошибка), MAE, Mean Absolute Error (средняя абсолютная ошибка), Huber Loss (функция Хьюбера), RMSE (корень из MSE), Quantile Loss (квантильная функция потерь), MAPE (средняя абсолютная процентная ошибка)
- **classification**: Categorical Cross-Entropy (категориальная кросс-энтропия), Hinge Loss (кусочно-линейная функция потерь), Focal Loss (фокальная функция потерь), Weighted Cross-Entropy (взвешенная кросс-энтропия), Gini / Entropy (критерий Джини / энтропия)
- **metric_learning**: Pairwise Loss (попарная функция потерь), Contrastive Loss (контрастивная функция потерь), Triplet Loss (тройная функция потерь), ArcFace / CosFace (угловые функции потерь), InfoNCE (контрастивная функция для многих негативов)
- **segmentation**: Dice Loss (функция потерь Дайса), IoU / Jaccard Loss (функция Жаккара)
- **generative**: Reconstruction Loss (ошибка восстановления), KL Divergence (расхождение Кульбака–Лейблера), Adversarial Loss (состязательная функция потерь), Diffusion Loss (функция потерь диффузии)
- **audio**: CTC Loss (Connectionist Temporal Classification), Mel-spectrogram Loss (ошибка по мел-спектрограмме)

</details>

<details><summary><b>optimizer</b> — 23</summary>

- **gradient_descent**: SGD (стохастический градиентный спуск), Mini-batch GD (спуск по мини-батчам)
- **momentum**: Momentum (импульс), Nesterov, NAG (ускоренный градиент Нестерова)
- **adaptive**: Adagrad (адаптивный градиент), RMSprop (среднеквадратичное распространение), Adadelta (ададельта)
- **adam_family**: Adam (адаптивная оценка моментов), AdamW (Adam с раздельным затуханием весов), Nadam (Adam с Нестеровым), AdaMax (адамакс), AMSGrad (амсград), RAdam (выпрямленный Adam)
- **large_scale**: LAMB (послойный адаптивный), Adafactor (адафактор), Lion (лайон)
- **second_order**: Newton's method (метод Ньютона), L-BFGS (квазиньютоновский с ограниченной памятью)
- **schedulers**: Step Decay (ступенчатое уменьшение), Exponential Decay (экспоненциальное уменьшение), Cosine Annealing (косинусный отжиг), Warmup (разогрев), Reduce on Plateau (снижение на плато)

</details>

<details><summary><b>metrics</b> — 35</summary>

- **regression**: MAE (средняя абсолютная ошибка), MSE (среднеквадратичная ошибка), RMSE (корень из MSE), R² (коэффициент детерминации), MAPE (средняя процентная ошибка)
- **classification**: Accuracy (доля правильных), Precision (точность), Recall (полнота), F1-score (F1-мера), Specificity (специфичность), Balanced Accuracy (сбалансированная доля правильных), MCC (коэффициент корреляции Мэтьюса), ROC-AUC (площадь под ROC-кривой), PR-AUC (площадь под кривой точность–полнота), confusion_matrix
- **ranking**: Precision@K (точность в топ-K), Recall@K (полнота в топ-K), MAP (средняя точность), MRR (средний обратный ранг), NDCG (нормализованный дисконтированный выигрыш)
- **clustering**: Silhouette (силуэт), Davies–Bouldin (индекс Дэвиса–Болдина), Calinski–Harabasz (индекс Калински–Харабаза), ARI (скорректированный индекс Рэнда), NMI (нормализованная взаимная информация)
- **vision**: IoU (пересечение над объединением), Dice (коэффициент Дайса), mAP (средняя точность для детекции)
- **generative**: Perplexity (перплексия), BLEU (блю), ROUGE (руж), FID (расстояние Фреше)
- **audio**: WER (доля ошибок по словам), CER (доля ошибок по символам), MOS (средняя оценка мнения)

</details>

## Проблемы

<details><summary><b><code>?</code> Нет строки <code>status</code> в шапке — 174</b></summary>

- Logistic Regression (логистическая регрессия) — [`logistic_regression.py`](src/model/supervised/classification/logistic_regression/logistic_regression.py)
- Histogram (гистограмма) — [`histogram.py`](src/data_prep/eda/distributions/histogram.py)
- KDE (оценка плотности ядром) — [`kde.py`](src/data_prep/eda/distributions/kde.py)
- Boxplot (ящик с усами) — [`boxplot.py`](src/data_prep/eda/distributions/boxplot.py)
- Pearson (корреляция Пирсона) — [`pearson.py`](src/data_prep/eda/correlations/pearson.py)
- Spearman (корреляция Спирмена) — [`spearman.py`](src/data_prep/eda/correlations/spearman.py)
- Cramér's V (V Крамера) — [`cramers_v.py`](src/data_prep/eda/correlations/cramers_v.py)
- Mutual Information (взаимная информация) — [`mutual_information.py`](src/data_prep/eda/correlations/mutual_information.py)
- Pairplot (попарные графики) — [`pairplot.py`](src/data_prep/eda/multivariate/pairplot.py)
- Heatmap (тепловая карта) — [`heatmap.py`](src/data_prep/eda/multivariate/heatmap.py)
- PCA projection (проекция PCA) — [`pca_projection.py`](src/data_prep/eda/multivariate/pca_projection.py)
- Missing report (отчёт о пропусках) — [`missing_report.py`](src/data_prep/preprocessing/missing/detection/missing_report.py)
- Missing pattern (структура пропусков) — [`missing_pattern.py`](src/data_prep/preprocessing/missing/detection/missing_pattern.py)
- Drop (удаление) — [`drop.py`](src/data_prep/preprocessing/missing/imputation/drop.py)
- Mean / Median / Mode (среднее / медиана / мода) — [`mean_median_mode.py`](src/data_prep/preprocessing/missing/imputation/mean_median_mode.py)
- Constant (константа) — [`constant.py`](src/data_prep/preprocessing/missing/imputation/constant.py)
- Interpolation (интерполяция) — [`interpolation.py`](src/data_prep/preprocessing/missing/imputation/interpolation.py)
- KNN Imputer (заполнение по соседям) — [`knn_imputer.py`](src/data_prep/preprocessing/missing/imputation/knn_imputer.py)
- Iterative Imputer, MICE (итеративное заполнение) — [`iterative_imputer.py`](src/data_prep/preprocessing/missing/imputation/iterative_imputer.py)
- Missing Indicator (индикатор пропуска) — [`missing_indicator.py`](src/data_prep/preprocessing/missing/imputation/missing_indicator.py)
- IQR (межквартильный размах) — [`iqr.py`](src/data_prep/preprocessing/outliers/detection/iqr.py)
- Z-score (стандартизованное отклонение) — [`z_score.py`](src/data_prep/preprocessing/outliers/detection/z_score.py)
- Modified Z-score (модифицированный Z) — [`modified_z_score.py`](src/data_prep/preprocessing/outliers/detection/modified_z_score.py)
- Isolation Forest (изолирующий лес) — [`isolation_forest.py`](src/data_prep/preprocessing/outliers/detection/isolation_forest.py)
- LOF (локальный фактор выброса) — [`lof.py`](src/data_prep/preprocessing/outliers/detection/lof.py)
- DBSCAN (кластеризация по плотности) — [`dbscan.py`](src/data_prep/preprocessing/outliers/detection/dbscan.py)
- Remove (удаление) — [`remove.py`](src/data_prep/preprocessing/outliers/treatment/remove.py)
- Clip / Winsorizing (обрезка) — [`clip.py`](src/data_prep/preprocessing/outliers/treatment/clip.py)
- Log transform (логарифм) — [`log_transform.py`](src/data_prep/preprocessing/outliers/treatment/log_transform.py)
- Impute (замена как пропуск) — [`impute.py`](src/data_prep/preprocessing/outliers/treatment/impute.py)
- Flag (пометка) — [`flag.py`](src/data_prep/preprocessing/outliers/treatment/flag.py)
- One-Hot (прямое кодирование) — [`one_hot.py`](src/data_prep/preprocessing/encoding/one_hot.py)
- Label (меточное) — [`label.py`](src/data_prep/preprocessing/encoding/label.py)
- Ordinal (порядковое) — [`ordinal.py`](src/data_prep/preprocessing/encoding/ordinal.py)
- Target (по целевой переменной) — [`target.py`](src/data_prep/preprocessing/encoding/target.py)
- Frequency (частотное) — [`frequency.py`](src/data_prep/preprocessing/encoding/frequency.py)
- Binary (двоичное) — [`binary.py`](src/data_prep/preprocessing/encoding/binary.py)
- Hashing (хеширование) — [`hashing.py`](src/data_prep/preprocessing/encoding/hashing.py)
- StandardScaler (стандартизация) — [`standard.py`](src/data_prep/preprocessing/scaling/standard.py)
- MinMaxScaler (минимакс) — [`minmax.py`](src/data_prep/preprocessing/scaling/minmax.py)
- RobustScaler (устойчивое масштабирование) — [`robust.py`](src/data_prep/preprocessing/scaling/robust.py)
- MaxAbsScaler (по максимуму модуля) — [`maxabs.py`](src/data_prep/preprocessing/scaling/maxabs.py)
- Normalizer (нормализация строк) — [`normalizer.py`](src/data_prep/preprocessing/scaling/normalizer.py)
- Log (логарифм) — [`log.py`](src/data_prep/preprocessing/transformation/log.py)
- Box-Cox (преобразование Бокса–Кокса) — [`box_cox.py`](src/data_prep/preprocessing/transformation/box_cox.py)
- Yeo-Johnson (преобразование Йео–Джонсона) — [`yeo_johnson.py`](src/data_prep/preprocessing/transformation/yeo_johnson.py)
- Quantile (квантильное) — [`quantile.py`](src/data_prep/preprocessing/transformation/quantile.py)
- Polynomial (полиномиальные) — [`polynomial.py`](src/data_prep/preprocessing/feature_engineering/polynomial.py)
- Interactions (взаимодействия) — [`interactions.py`](src/data_prep/preprocessing/feature_engineering/interactions.py)
- Binning (разбиение на интервалы) — [`binning.py`](src/data_prep/preprocessing/feature_engineering/binning.py)
- Datetime features (признаки из даты) — [`datetime_features.py`](src/data_prep/preprocessing/feature_engineering/datetime_features.py)
- Aggregations (агрегаты) — [`aggregations.py`](src/data_prep/preprocessing/feature_engineering/aggregations.py)
- Variance Threshold (порог дисперсии) — [`variance_threshold.py`](src/data_prep/preprocessing/feature_selection/filter/variance_threshold.py)
- Correlation (корреляция) — [`correlation.py`](src/data_prep/preprocessing/feature_selection/filter/correlation.py)
- Chi² (хи-квадрат) — [`chi2.py`](src/data_prep/preprocessing/feature_selection/filter/chi2.py)
- Mutual Information (взаимная информация) — [`mutual_information.py`](src/data_prep/preprocessing/feature_selection/filter/mutual_information.py)
- RFE (рекурсивное исключение признаков) — [`rfe.py`](src/data_prep/preprocessing/feature_selection/wrapper/rfe.py)
- Forward Selection (прямой отбор) — [`forward_selection.py`](src/data_prep/preprocessing/feature_selection/wrapper/forward_selection.py)
- Backward Elimination (обратное исключение) — [`backward_elimination.py`](src/data_prep/preprocessing/feature_selection/wrapper/backward_elimination.py)
- Lasso (L1-регуляризация) — [`lasso.py`](src/data_prep/preprocessing/feature_selection/embedded/lasso.py)
- Tree Importance (важность по дереву) — [`tree_importance.py`](src/data_prep/preprocessing/feature_selection/embedded/tree_importance.py)
- Holdout (отложенная выборка) — [`holdout.py`](src/data_prep/preprocessing/splitting/holdout.py)
- Stratified (стратифицированное) — [`stratified.py`](src/data_prep/preprocessing/splitting/stratified.py)
- K-Fold (k-кратная кросс-валидация) — [`kfold.py`](src/data_prep/preprocessing/splitting/kfold.py)
- Stratified K-Fold (стратифицированная k-кратная) — [`stratified_kfold.py`](src/data_prep/preprocessing/splitting/stratified_kfold.py)
- Group K-Fold (k-кратная по группам) — [`group_kfold.py`](src/data_prep/preprocessing/splitting/group_kfold.py)
- Time Series Split (разбиение по времени) — [`time_series_split.py`](src/data_prep/preprocessing/splitting/time_series_split.py)
- Leave-One-Out (исключение по одному) — [`leave_one_out.py`](src/data_prep/preprocessing/splitting/leave_one_out.py)
- Random Oversampling (случайное дублирование) — [`random_oversampling.py`](src/data_prep/preprocessing/imbalance/oversampling/random_oversampling.py)
- SMOTE (синтетическое дополнение) — [`smote.py`](src/data_prep/preprocessing/imbalance/oversampling/smote.py)
- ADASYN (адаптивное синтетическое) — [`adasyn.py`](src/data_prep/preprocessing/imbalance/oversampling/adasyn.py)
- Random Undersampling (случайное удаление) — [`random_undersampling.py`](src/data_prep/preprocessing/imbalance/undersampling/random_undersampling.py)
- Tomek Links (связи Томека) — [`tomek_links.py`](src/data_prep/preprocessing/imbalance/undersampling/tomek_links.py)
- Class Weights (веса классов) — [`class_weights.py`](src/data_prep/preprocessing/imbalance/class_weights.py)
- Identity / Linear (тождественная) — [`identity.py`](src/activations/output/identity.py)
- Sigmoid (сигмоида) — [`sigmoid.py`](src/activations/output/sigmoid.py)
- Softmax (софтмакс) — [`softmax.py`](src/activations/output/softmax.py)
- Tanh (гиперболический тангенс) — [`tanh.py`](src/activations/output/tanh.py)
- Sigmoid (сигмоида) — [`sigmoid.py`](src/activations/hidden/saturating/sigmoid.py)
- Tanh (гиперболический тангенс) — [`tanh.py`](src/activations/hidden/saturating/tanh.py)
- ReLU (выпрямитель) — [`relu.py`](src/activations/hidden/relu_family/relu.py)
- Leaky ReLU (протекающий ReLU) — [`leaky_relu.py`](src/activations/hidden/relu_family/leaky_relu.py)
- PReLU (параметрический ReLU) — [`prelu.py`](src/activations/hidden/relu_family/prelu.py)
- ELU (экспоненциальный линейный) — [`elu.py`](src/activations/hidden/relu_family/elu.py)
- SELU (масштабированный ELU) — [`selu.py`](src/activations/hidden/relu_family/selu.py)
- GELU (гауссов линейный) — [`gelu.py`](src/activations/hidden/smooth/gelu.py)
- Swish / SiLU (свиш) — [`swish.py`](src/activations/hidden/smooth/swish.py)
- Mish (миш) — [`mish.py`](src/activations/hidden/smooth/mish.py)
- Softplus (мягкий плюс) — [`softplus.py`](src/activations/hidden/smooth/softplus.py)
- MSE, Mean Squared Error (среднеквадратичная ошибка) — [`mse.py`](src/losses/regression/mse.py)
- MAE, Mean Absolute Error (средняя абсолютная ошибка) — [`mae.py`](src/losses/regression/mae.py)
- Huber Loss (функция Хьюбера) — [`huber.py`](src/losses/regression/huber.py)
- RMSE (корень из MSE) — [`rmse.py`](src/losses/regression/rmse.py)
- Quantile Loss (квантильная функция потерь) — [`quantile.py`](src/losses/regression/quantile.py)
- MAPE (средняя абсолютная процентная ошибка) — [`mape.py`](src/losses/regression/mape.py)
- Log Loss / Binary Cross-Entropy (бинарная кросс-энтропия) — [`log_loss.py`](src/losses/classification/log_loss.py)
- Categorical Cross-Entropy (категориальная кросс-энтропия) — [`cross_entropy.py`](src/losses/classification/cross_entropy.py)
- Hinge Loss (кусочно-линейная функция потерь) — [`hinge.py`](src/losses/classification/hinge.py)
- Focal Loss (фокальная функция потерь) — [`focal.py`](src/losses/classification/focal.py)
- Weighted Cross-Entropy (взвешенная кросс-энтропия) — [`weighted_cross_entropy.py`](src/losses/classification/weighted_cross_entropy.py)
- Gini / Entropy (критерий Джини / энтропия) — [`gini_entropy.py`](src/losses/classification/gini_entropy.py)
- Pairwise Loss (попарная функция потерь) — [`pairwise.py`](src/losses/metric_learning/pairwise.py)
- Contrastive Loss (контрастивная функция потерь) — [`contrastive.py`](src/losses/metric_learning/contrastive.py)
- Triplet Loss (тройная функция потерь) — [`triplet.py`](src/losses/metric_learning/triplet.py)
- ArcFace / CosFace (угловые функции потерь) — [`arcface.py`](src/losses/metric_learning/arcface.py)
- InfoNCE (контрастивная функция для многих негативов) — [`info_nce.py`](src/losses/metric_learning/info_nce.py)
- Dice Loss (функция потерь Дайса) — [`dice.py`](src/losses/segmentation/dice.py)
- IoU / Jaccard Loss (функция Жаккара) — [`iou.py`](src/losses/segmentation/iou.py)
- Reconstruction Loss (ошибка восстановления) — [`reconstruction.py`](src/losses/generative/reconstruction.py)
- KL Divergence (расхождение Кульбака–Лейблера) — [`kl_divergence.py`](src/losses/generative/kl_divergence.py)
- Adversarial Loss (состязательная функция потерь) — [`adversarial.py`](src/losses/generative/adversarial.py)
- Diffusion Loss (функция потерь диффузии) — [`diffusion.py`](src/losses/generative/diffusion.py)
- CTC Loss (Connectionist Temporal Classification) — [`ctc.py`](src/losses/audio/ctc.py)
- Mel-spectrogram Loss (ошибка по мел-спектрограмме) — [`mel_spectrogram.py`](src/losses/audio/mel_spectrogram.py)
- Batch GD (полный градиентный спуск) — [`batch_gd.py`](src/optimizers/gradient_descent/batch_gd.py)
- SGD (стохастический градиентный спуск) — [`sgd.py`](src/optimizers/gradient_descent/sgd.py)
- Mini-batch GD (спуск по мини-батчам) — [`minibatch_gd.py`](src/optimizers/gradient_descent/minibatch_gd.py)
- Momentum (импульс) — [`momentum.py`](src/optimizers/momentum/momentum.py)
- Nesterov, NAG (ускоренный градиент Нестерова) — [`nesterov.py`](src/optimizers/momentum/nesterov.py)
- Adagrad (адаптивный градиент) — [`adagrad.py`](src/optimizers/adaptive/adagrad.py)
- RMSprop (среднеквадратичное распространение) — [`rmsprop.py`](src/optimizers/adaptive/rmsprop.py)
- Adadelta (ададельта) — [`adadelta.py`](src/optimizers/adaptive/adadelta.py)
- Adam (адаптивная оценка моментов) — [`adam.py`](src/optimizers/adam_family/adam.py)
- AdamW (Adam с раздельным затуханием весов) — [`adamw.py`](src/optimizers/adam_family/adamw.py)
- Nadam (Adam с Нестеровым) — [`nadam.py`](src/optimizers/adam_family/nadam.py)
- AdaMax (адамакс) — [`adamax.py`](src/optimizers/adam_family/adamax.py)
- AMSGrad (амсград) — [`amsgrad.py`](src/optimizers/adam_family/amsgrad.py)
- RAdam (выпрямленный Adam) — [`radam.py`](src/optimizers/adam_family/radam.py)
- LAMB (послойный адаптивный) — [`lamb.py`](src/optimizers/large_scale/lamb.py)
- Adafactor (адафактор) — [`adafactor.py`](src/optimizers/large_scale/adafactor.py)
- Lion (лайон) — [`lion.py`](src/optimizers/large_scale/lion.py)
- Newton's method (метод Ньютона) — [`newton.py`](src/optimizers/second_order/newton.py)
- L-BFGS (квазиньютоновский с ограниченной памятью) — [`lbfgs.py`](src/optimizers/second_order/lbfgs.py)
- Step Decay (ступенчатое уменьшение) — [`step_decay.py`](src/optimizers/schedulers/step_decay.py)
- Exponential Decay (экспоненциальное уменьшение) — [`exponential_decay.py`](src/optimizers/schedulers/exponential_decay.py)
- Cosine Annealing (косинусный отжиг) — [`cosine_annealing.py`](src/optimizers/schedulers/cosine_annealing.py)
- Warmup (разогрев) — [`warmup.py`](src/optimizers/schedulers/warmup.py)
- Reduce on Plateau (снижение на плато) — [`reduce_on_plateau.py`](src/optimizers/schedulers/reduce_on_plateau.py)
- MAE (средняя абсолютная ошибка) — [`mae.py`](src/metrics/regression/mae.py)
- MSE (среднеквадратичная ошибка) — [`mse.py`](src/metrics/regression/mse.py)
- RMSE (корень из MSE) — [`rmse.py`](src/metrics/regression/rmse.py)
- R² (коэффициент детерминации) — [`r2.py`](src/metrics/regression/r2.py)
- MAPE (средняя процентная ошибка) — [`mape.py`](src/metrics/regression/mape.py)
- Accuracy (доля правильных) — [`accuracy.py`](src/metrics/classification/accuracy.py)
- Precision (точность) — [`precision.py`](src/metrics/classification/precision.py)
- Recall (полнота) — [`recall.py`](src/metrics/classification/recall.py)
- F1-score (F1-мера) — [`f1.py`](src/metrics/classification/f1.py)
- Specificity (специфичность) — [`specificity.py`](src/metrics/classification/specificity.py)
- Balanced Accuracy (сбалансированная доля правильных) — [`balanced_accuracy.py`](src/metrics/classification/balanced_accuracy.py)
- MCC (коэффициент корреляции Мэтьюса) — [`mcc.py`](src/metrics/classification/mcc.py)
- ROC-AUC (площадь под ROC-кривой) — [`roc_auc.py`](src/metrics/classification/roc_auc.py)
- PR-AUC (площадь под кривой точность–полнота) — [`pr_auc.py`](src/metrics/classification/pr_auc.py)
- Precision@K (точность в топ-K) — [`precision_at_k.py`](src/metrics/ranking/precision_at_k.py)
- Recall@K (полнота в топ-K) — [`recall_at_k.py`](src/metrics/ranking/recall_at_k.py)
- MAP (средняя точность) — [`map.py`](src/metrics/ranking/map.py)
- MRR (средний обратный ранг) — [`mrr.py`](src/metrics/ranking/mrr.py)
- NDCG (нормализованный дисконтированный выигрыш) — [`ndcg.py`](src/metrics/ranking/ndcg.py)
- Silhouette (силуэт) — [`silhouette.py`](src/metrics/clustering/silhouette.py)
- Davies–Bouldin (индекс Дэвиса–Болдина) — [`davies_bouldin.py`](src/metrics/clustering/davies_bouldin.py)
- Calinski–Harabasz (индекс Калински–Харабаза) — [`calinski_harabasz.py`](src/metrics/clustering/calinski_harabasz.py)
- ARI (скорректированный индекс Рэнда) — [`ari.py`](src/metrics/clustering/ari.py)
- NMI (нормализованная взаимная информация) — [`nmi.py`](src/metrics/clustering/nmi.py)
- IoU (пересечение над объединением) — [`iou.py`](src/metrics/vision/iou.py)
- Dice (коэффициент Дайса) — [`dice.py`](src/metrics/vision/dice.py)
- mAP (средняя точность для детекции) — [`map_detection.py`](src/metrics/vision/map_detection.py)
- Perplexity (перплексия) — [`perplexity.py`](src/metrics/generative/perplexity.py)
- BLEU (блю) — [`bleu.py`](src/metrics/generative/bleu.py)
- ROUGE (руж) — [`rouge.py`](src/metrics/generative/rouge.py)
- FID (расстояние Фреше) — [`fid.py`](src/metrics/generative/fid.py)
- WER (доля ошибок по словам) — [`wer.py`](src/metrics/audio/wer.py)
- CER (доля ошибок по символам) — [`cer.py`](src/metrics/audio/cer.py)
- MOS (средняя оценка мнения) — [`mos.py`](src/metrics/audio/mos.py)
- ffill_bfill — [`ffill_bfill.py`](src/data_prep/preprocessing/missing/imputation/ffill_bfill.py)
- confusion_matrix — [`confusion_matrix.py`](src/metrics/classification/confusion_matrix.py)

</details>

**Нет файла: slug не совпал ни с одним файлом или папкой — 21** (впишите правильный `slug` в таблицу)

- k-NN (k ближайших соседей) — `model/supervised/classification`
- Ridge / Lasso (регрессия с регуляризацией) — `model/supervised/regression`
- k-NN — `model/supervised/regression`
- бустинг с ranking loss — `model/supervised/ranking`
- K-Means — `model/unsupervised/clustering`
- DBSCAN / HDBSCAN (по плотности) — `model/unsupervised/clustering`
- Spectral Clustering (спектральная) — `model/unsupervised/clustering`
- t-SNE — `model/unsupervised/dimensionality_reduction`
- автоэнкодер по ошибке восстановления — `model/unsupervised/anomaly_detection`
- Image Classification (классификация изображений) — `model/deep_learning`
- Object Detection (детекция объектов) — `model/deep_learning`
- Segmentation (сегментация) — `model/deep_learning`
- Face Recognition (распознавание лиц) — `model/deep_learning`
- ASR, Speech Recognition (распознавание речи) — `model/deep_learning`
- TTS, Text-to-Speech (синтез речи) — `model/deep_learning`
- NLP: Text Classification (классификация текста) — `model/deep_learning`
- Seq2Seq: Machine Translation (машинный перевод) — `model/deep_learning`
- Generative models (генеративные модели) — `model/deep_learning`
- Time Series Forecasting (прогноз временных рядов) — `model/deep_learning`
- Recommender Systems (рекомендательные системы) — `model/deep_learning`
- Forward / Backward fill (заполнение соседним) — `analysis/preprocessing/missing`
