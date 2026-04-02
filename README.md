# GennadyObjectLearningModel

## Что это за проект
`GennadyObjectLearningModel` — модульный CV-конвейер для обработки фото и видео с людьми. Текущий пайплайн включает: детекцию, извлечение позы, human parsing, трекинг, сборку сцены, рендер отладочных изображений и сохранение результатов в `output/`.

## Архитектура пайплайна
1. **Detector**: находит людей и возвращает `bbox`.
2. **PoseExtractor**: извлекает ключевые точки по детекциям.
3. **HumanParser**: сегментирует человека по классам.
   - `segformer` → schema `v1` (legacy labels одежды/частей).
   - `sam2` → schema `sam2` (реальная SAM2 сегментация человека + coarse inferred parts).
4. **Tracker**: сопоставляет людей между кадрами и переносит parsing при пропусках.
5. **SceneBuilder**: собирает `SceneFrame`.
6. **Renderer**: генерирует визуализации.
7. **OutputWriter**: сохраняет PNG-файлы.

Оркестратор (`PipelineOrchestrator`) разделяет быстрый контур (detector+pose) и медленный асинхронный контур сегментации.

## HumanRepresentation (dual-schema)
Добавлен логический слой `src/representation/`, который строит единое нормализованное представление человека поверх разных схем parsing без замены существующего flow.

`HumanRepresentation v1` содержит:
- `human_id`, `bbox`, `person_mask`
- именованные `keypoints`
- `body_parts`
- `garments`
- `relations`
- `state`
- `reliability/evidence` поля у body parts и garments
- агрегаты `dominant_garments`, `has_layered_upper_clothing`, счетчики надежных сущностей
- технические поля кадра и confidence

Все координаты и маски остаются в координатах исходного кадра.

## Сущности HumanRepresentation
Основные dataclass-сущности:
- `MaskRegion`
- `Keypoint2D`
- `BodyPart`
- `Garment`
- `RelationEdge`
- `HumanState`
- `HumanRepresentation`

Поддерживаются типы:
- части тела (`head`, `face`, `torso`, `left_arm`, ...)
- типы одежды (`upper_inner`, `pants`, `shoes`, ...)
- связи (`attached_to`, `covers`, ...)
- состояния (`standing/sitting/lying`, состояния рук)

## Как builder собирает representation
`HumanRepresentationBuilder` использует `TrackedHuman` и schema-aware lifting:
- `human_id` = `human_{track_id}`
- `bbox` из `detection`
- `person_mask` как union всех parsing-масок (если parsing есть)
- `person_mask`, body parts и garments мягко клипаются к bbox для более консистентного представления
- `keypoints` из `PoseResult` с аккуратным именованием MediaPipe-индексов
- `schema v1`, `schema v2` и `schema sam2` обрабатываются отдельными адаптерами (`src/representation/parsing_adapters.py`)
- `v2` строит и fine anatomy (`left_upper_arm`, `left_thigh`, `back_upper`, ...) и coarse body parts (`torso`, `left_arm`, `left_leg`, ...)
- `sam2` строит `person_mask` из реального вывода модели SAM2 и добавляет coarse части (`head/torso/left_arm/right_arm/left_leg/right_leg`) как **heuristic inferred**
- на `schema sam2` garments сейчас **не создаются автоматически**, чтобы не выдавать эвристику за факт
- `garments` в `v2` строятся через anatomy anchors (torso/arms/legs) + image cues для layered upper (`upper_inner`/`outerwear`)
- `reliability_score` + категориальная `reliability` по visibility/evidence/типу источников
- normalization/filtering: слабые сущности подавляются в overlay/relations, но сохраняются для debug
- `relations` rule-based c учетом reliability и suppression
- `state` (pose/arms) по геометрическим правилам

Если `pose` или `parsing` отсутствуют, builder возвращает валидный объект без падения.
При нехватке данных поведение консервативное: неуверенные регионы/связи могут не создаваться.

## Ограничения и статус интеграции
- Логика intentionally эвристическая (без дополнительных нейросетей).
- Тип одежды в сложных случаях может упрощаться до `pants`/`upper_inner`/`unknown_garment` (для схем `v1`/`v2`).
- Руки и шея строятся эвристически и могут отсутствовать при нехватке надежных keypoints.
- Если у парсинга есть только общая `shoes`-маска, `left_foot/right_foot` аппроксимируются делением маски по центру bbox.
- Layered upper clothing строится эвристически и гибридно (контраст/центр/геометрия), а не learned-моделью.
- `sam2` backend теперь использует **реальный prompt-based inference** через установленный пакет `sam2`.
- SAM2 **не входит** в обязательные зависимости `pyproject.toml` и должен быть установлен отдельно (обычно из official/local репозитория SAM2).
- Этот проект ожидает, что в окружении доступны импорты:
  - `sam2.build_sam`
  - `sam2.sam2_image_predictor`
- SAM2 в текущем контуре предсказывает **mask человека**, а не fine anatomy labels.
- Fine anatomy (`chest_left`, `pelvis`, `upper_arm_left` и т.п.) из SAM2 **не выдумывается**.
- Coarse части на SAM2 path помечены как `heuristic`/`inferred_only`.
- Garment semantics на `schema sam2` автоматически не выводятся.
- В weak-кейсах применено консервативное поведение: система откатывается к одному `upper_inner` без принудительного `outerwear`.
- Связи (`relations`) и `occlusion` теперь reliability-aware: низконадежные сущности чаще исключаются из relation-слоя.
- Временная устойчивость пока на уровне groundwork через стабильный `human_id` в треке.

## Как запустить проект
```bash
python main.py --use-mock
```
или рабочий режим с моделями:
```bash
python main.py --device cpu
```
SAM2 backend (Linux/WSL, CUDA):
```bash
python main.py --device cuda --parser-backend sam2 --sam2-checkpoint /absolute/path/to/sam2_checkpoint.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --sam2-use-pose-prompts
```
Предустановка SAM2 (пример):
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Опциональные аргументы:
- `--parsing-interval` (по умолчанию `5`)
- `--device {cpu,cuda}`
- `--parser-backend {segformer,sam2}`
- `--use-mock`
- `--sam2-checkpoint` (или env `SAM2_CHECKPOINT`) — обязательный путь для `--parser-backend sam2`
- `--sam2-config` (или env `SAM2_MODEL_CFG`) — имя/путь конфигурации SAM2
- `--sam2-device {cpu,cuda}` — устройство для SAM2 (по умолчанию как `--device`)
- `--sam2-use-pose-prompts` — добавлять pose keypoints как positive prompts

## Какие артефакты сохраняются в output
Для каждого сохраняемого кадра пишутся PNG:
- `output/skeleton/`
- `output/parsing/`
- `output/detection/`
- `output/combined/`
- `output/representation_overlay/`
- `output/representation_debug/`
- `output/representation_masks/`
- `output/representation_masks_raw/`
- `output/representation_masks_normalized/`
- `output/representation_masks_garments/`
- `output/anatomy_raw_overlay/`
- `output/sam2_raw_mask/`
- `output/sam2_prompt_debug/`
- `output/summary_panel/`

Имена файлов стабильные: `<base_name>_frame_<индекс>.png`.

## Что смотреть в PNG для быстрой оценки
1. `combined` — базовая согласованность detection/pose/parsing.
2. `representation_overlay` — компактный reliability-aware слой (слои одежды + число надежных сущностей).
3. `anatomy_raw_overlay` — сырые labels схемы текущего backend.
4. `representation_masks_raw` — максимально подробные body-part маски (включая fine anatomy и подавленные регионы бледно).
5. `representation_masks_normalized` — нормализованные coarse части тела для reasoning-уровня.
6. `representation_masks_garments` — garment semantics (наиболее информативно для `v1`/`v2`; для `sam2` может быть пусто).
7. `representation_debug` — подробные списки частей тела/одежды/связей/confidence/evidence/suppression-флагов.
8. `summary_panel` — 3x3 сводка: detection/parsing/raw anatomy/normalized/garments/overlay/debug.
9. `sam2_raw_mask` — отдельная проверка реальной person mask от SAM2.
10. `sam2_prompt_debug` — prompt box/points, отправленные в SAM2.

## Следующий этап развития
- Улучшение стабильности garment-идентичности во времени.
- Более точная infer-логика состояний и окклюзий.
- Добавление правил для multi-person взаимодействий и richer relations.
- Введение метрик качества representation и регрессионных тестов на датасетах.
