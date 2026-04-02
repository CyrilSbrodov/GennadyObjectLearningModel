# GennadyObjectLearningModel

## Что это за проект
`GennadyObjectLearningModel` — модульный CV-конвейер для обработки фото и видео с людьми. Текущий пайплайн включает: детекцию, извлечение позы, human parsing, трекинг, сборку сцены, рендер отладочных изображений и сохранение результатов в `output/`.

## Архитектура пайплайна
1. **Detector**: находит людей и возвращает `bbox`.
2. **PoseExtractor**: извлекает ключевые точки по детекциям.
3. **HumanParser**: сегментирует человека по классам (face/hair/upper/lower/...)
4. **Tracker**: сопоставляет людей между кадрами и переносит parsing при пропусках.
5. **SceneBuilder**: собирает `SceneFrame`.
6. **Renderer**: генерирует визуализации.
7. **OutputWriter**: сохраняет PNG-файлы.

Оркестратор (`PipelineOrchestrator`) разделяет быстрый контур (detector+pose) и медленный асинхронный контур сегментации.

## HumanRepresentation v1
Добавлен новый логический слой `src/representation/`, который строит структурированное представление человека поверх текущих сенсоров без замены существующего flow.

`HumanRepresentation v1` содержит:
- `human_id`, `bbox`, `person_mask`
- именованные `keypoints`
- `body_parts`
- `garments`
- `relations`
- `state`
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
`HumanRepresentationBuilder` использует `TrackedHuman`:
- `human_id` = `human_{track_id}`
- `bbox` из `detection`
- `person_mask` как union всех parsing-масок (если parsing есть)
- `person_mask`, body parts и garments мягко клипаются к bbox для более консистентного v1-представления
- `keypoints` из `PoseResult` с аккуратным именованием MediaPipe-индексов
- `body_parts` из parsing + простых эвристик
- `garments` из mapping parsing labels
- `relations` rule-based между garment и body parts
- `state` (pose/arms) по геометрическим правилам

Если `pose` или `parsing` отсутствуют, builder возвращает валидный объект без падения.
При нехватке данных поведение консервативное: неуверенные регионы/связи могут не создаваться.

## Ограничения v1
- Логика intentionally эвристическая (без дополнительных нейросетей).
- Тип одежды в сложных случаях может упрощаться до `pants`/`upper_inner`/`unknown_garment`.
- Руки и шея строятся эвристически и могут отсутствовать при нехватке надежных keypoints.
- Если у парсинга есть только общая `shoes`-маска, `left_foot/right_foot` аппроксимируются делением маски по центру bbox.
- Layering одежды в v1 ограничен базовыми правилами (`outerwear` поверх `upper_inner`, если оба слоя есть).
- Связи (`relations`) и `occlusion` в v1 базовые и консервативные: лучше меньше связей, чем уверенно ложные.
- Временная устойчивость пока на уровне groundwork через стабильный `human_id` в треке.

## Как запустить проект
```bash
python main.py --use-mock
```
или рабочий режим с моделями:
```bash
python main.py --device cpu
```

Опциональные аргументы:
- `--parsing-interval` (по умолчанию `5`)
- `--device {cpu,cuda}`
- `--use-mock`

## Какие артефакты сохраняются в output
Для каждого сохраняемого кадра пишутся PNG:
- `output/skeleton/`
- `output/parsing/`
- `output/detection/`
- `output/combined/`
- `output/representation_overlay/`
- `output/representation_debug/`
- `output/summary_panel/`

Имена файлов стабильные: `<base_name>_frame_<индекс>.png`.

## Что смотреть в PNG для быстрой оценки
1. `combined` — базовая согласованность detection/pose/parsing.
2. `representation_overlay` — компактный слой (bbox + ключевые labels + краткое состояние), без перегрузки кадра.
3. `representation_debug` — подробные списки частей тела/одежды/связей/confidence.
4. `summary_panel` — быстрая сравнительная сводка, где debug-preview адаптирован для лучшей читаемости текста.

## Следующий этап развития
- Улучшение стабильности garment-идентичности во времени.
- Более точная infer-логика состояний и окклюзий.
- Добавление правил для multi-person взаимодействий и richer relations.
- Введение метрик качества representation и регрессионных тестов на датасетах.
