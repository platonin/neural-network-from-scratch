# neural-network-from-scratch
*Цель проекта – создание полносвязной нейронной сети с нуля.*

---
### Установка
1. Клонируйте репозиторий с субмодулями:
```bash
git clone --recurse-submodules https://github.com/platonin/neural-network-from-scratch
```

2. Установите Qt6, если не установлен (https://doc.qt.io/qt-6/qt-online-installation.html)
Для MacOs можно установить через brew, выполнив команду:
```bash
brew install qt
```

3. Выполните сборку проекта:
```bash
mkdir build
cd build
cmake ..
make
```

- Для запуска консольного приложения:
```bash
./console_app
```
- Для запуска примера использования библиотеки (main.cpp):
```bash
./main
```
- Для запуска примера загрузки сохраненной нейросети и проверки точности (main_check.cpp):
```bash
./main_check
```

---
### Что реализовано?

- **Оптимизаторы:**
  - SGD
  - SGD + momentum
  - RMSProp
  - Adam
- **Функции активации:**
  - Sigmoid
  - ReLU
  - Leaky ReLU
  - Softmax
- **Функции потерь:**
  - MSE
  - MAE
  - Cross-entropy

### Как пользоваться библиотекой
#### Создание экземпляра нейросети
При создании надо указать:
- размер входного вектора
- функцию потерь
- параметры слоев (их количество, размеры выходных векторов, функции активации для каждого слоя)
- оптимизатор

**Пример:**
```cpp
ActivationFunc sigmoid = ActivationCreation::GetSigmod();
ActivationFunc softmax = ActivationCreation::GetSoftmax();
NetBuilder builder(784); // задаем входной размер при создании
builder.setLoss(LossCreation::GetCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
builder.setLayers({{128, sigmoid}, {64, sigmoid}, {10, softmax}}); // добавляем слои с функциями активации
builder.setOptimizer(OptimizerCreation::GetAdam(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum
Net net = builder.createNet();
```
#### Обучение нейросети

После создания нейросети с помощью NetBuilder надо вызвать метод `train(X, Y, epochs, batchSize)` у экземпляра класса Net

`train` - выполняет обучение нейронной сети с использованием указанного в builder оптимизатора. Принимает обучающие данные X и Y, количество эпох epochs, скорость обучения learningRate и размер батча batchSize, и выводит прогресс обучения и точность на тренировочной выборке.

#### Сохранение нейросети

Чтобы сохранить параметры нейросети (чтобы потом можно было использовать уже обученную нейросеть) есть метод `SaveNet("path")`, который сохраняет веса в файл `temporary_weights.txt` по указанному пути.

---

- Пример обучения для распознавания цифр из MNIST можно посмотреть в `main.cpp`.
- Пример загрузки нейросети из файла и проверки точности распознования цифр из MNIST можно в `main_check.cpp`.

---

## Результаты обучения

Пока что самой удачной конфигурацией получилось добиться 97,14% точности.

Конфигурация:****
- Слой 1:
  - входной размер: 784
  - выходной размер: 128
  - функция активации: sigmoid
- Слой 2:
  - входной размер: 128
  - выходной размер: 64
  - функция активации: sigmoid
- Слой 3:
  - входной размер: 64
  - выходной размер: 10
  - функция активации: softmax
- функция потерь: cross-entropy
- оптимизатор: adam

Обучение:
- размер батча: 128
- количество эпох: 7
- обучение происходило по всей выборке (60000 изображений)

```cpp
// Создание нейросети
ActivationFunc softmax = ActivationCreation::getSoftmax();
NeuralNetwork::NetBuilder builder(784);
builder.setLoss(LossCreation::getCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
builder.setLayers({{128, sigmoid}, {64, sigmoid}, {10, softmax}}); // добавляем слои с функциями активации
builder.setOptimizer(OptimizerCreation::getAdam(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum
NeuralNetwork::Net net = builder.createNet(); 

// Обучение
net.train(X, Y, 7, 128);
```
