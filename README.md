# FERMATRICA_UTILS

Универсальный функционал для FERMATRICA.

### 1. Ключевые идеи

В этом репозитории собраны наиболее базовый / фундаментальный функционал, не являющийся специфичным для моделирования маркетингового микса и в принципе двухконтурного моделирования временных рядов / панельных моделей.

Функционал FERMATRICA_UTILS может использоваться как в рамках фреймворка FERMATRICA, так и вне его.

### 2. Состав

- Работа с операционной системой / файлами и папками (`fermatrica_utils.os`) 
- Работа с объектами / классами (`fermatrica_utils.objects`)
- Динамическое / нестандартное выполнение кода (`fermatrica_utils.flow`)
- Массивы (включая фреймы и серии `pandas`) (`fermatrica_utils.arrays`)
- Математические функции и объекты (`fermatrica_utils.math`)
- Данные / датасеты (что-то более специфичное, чем массивы) (`fermatrica_utils.data`)

### 3. Установка

Для корректной работы рекомендуется установить все составляющие фреймворка FERMATRICA. Предполагается, что работа будет вестись в PyCharm

1. Создайте виртуальную среду Python удобного для вас типа (Anaconda, Poetry и т.п.) или воспользуйтесь ранее созданной. Имеет смысл завести отдельную виртуальную среду для эконометрических задач и для каждой новой версии FERMATRICA
   - Мини-гайд по виртуальным средам (внешний): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
   - Для версии фреймворка v010 пусть виртуальная среда называется FERMATRICA_v010
2. Клонируйте в удобное для вас место репозитории FERMATRICA
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
    - Для работы с интерактивным дашбордом клонируйте и его репозиторий
   ```commandline
   git clone https://github.com/FERMATRICA/fermatrica_dash.git 
   ```
   - Для предварительной работы с данными
   ```commandline
   git clone https://github.com/FERMATRICA/fermatrica_data.git
   ```
3. В каждом из репозиториев выберите среду FERMATRICA_v010 (FERMATRICA_v020, FERMATRICA_v030 и т.д.) через `Add interpreter` (в интерфейсе PyCharm) и переключитесь в соответствующую ветку гита
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    git checkout v010 [v020, v030...]
    ```
4. Установите все склонированные пакеты, кроме FERMATRICA_DASH, используя `pip install`
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   - Вместо перехода в папку каждого проекта можно указать путь к нему в `pip install`
   ```commandline
   pip install [path_to_fermatrica_part]
   ```
5. При необходимости, поставьте сторонние пакеты / библиотеки Python, которые требуются для функционирования FERMATRICA, используя `conda install` или `pip install`. Для обновления версий сторонних пакетов используйте `conda update` или `pip install -U`



