# FERMATRICA_UTILS

[_Russian version below_](#RU)

Universal functionality for FERMATRICA.

### 1. Key Ideas
This repository contains the most basic/fundamental functionality that is not specific to marketing mix modeling and, in general, two-contour modeling of time series/panel models.

The functionality of FERMATRICA_UTILS can be used both within the FERMATRICA framework and independently of it.

### 2. Components

- Working with the operating system/files and folders (`fermatrica_utils.os`)
- Working with objects/classes (`fermatrica_utils.objects`)
- Dynamic/non-standard code execution (`fermatrica_utils.flow`)
- Arrays (including pandas data frames and series) (`fermatrica_utils.arrays`)
- Mathematical functions and objects (`fermatrica_utils.math`)
- Data/datasets (something more specific than arrays) (`fermatrica_utils.data`)

### 3. Installation

To facilitate work, it is recommended to install all components of the FERMATRICA framework. It is assumed that work will be conducted in PyCharm (VScode is OK also for sure).

1. Create a Python virtual environment of your choice (Anaconda, Poetry, etc.) or use a previously created one. It makes sense to establish a separate virtual environment for econometric tasks and for every new version of FERMATRICA.
    1. Mini-guide on virtual environments (external): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
    2. For framework version v010, let the virtual environment be named FERMATRICA_v010.
2. Clone the FERMATRICA repositories to a location of your choice.
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
   1. To work with the interactive dashboard: _coming soon_
    2. For preliminary data work: _coming soon_
3. In each of the repositories, select the FERMATRICA_v010 environment (FERMATRICA_v020, FERMATRICA_v030, etc.) through Add interpreter in the PyCharm interface and switch to the corresponding git branch.
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    git checkout v010 [v020, v030...]
    ```
4. Install all cloned packages except FERMATRICA_DASH using pip install.
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   1. Instead of navigating to each project's folder, you can specify the path to it in pip install:
       ```commandline
       pip install [path_to_fermatrica_part]
       ```
5. If necessary, install third-party packages/libraries required for the functioning of FERMATRICA using `conda install` or `pip install`. To update versions of third-party packages, use `conda update` or `pip install -U`.

-------------------------------------

<a name="RU"></a>
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
    - Для работы с интерактивным дашбордом: _coming soon_
   - Для предварительной работы с данными: _coming soon_
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



