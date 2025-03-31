# neural-network-from-scratch
*Цель проекта – создание полносвязной нейронной сети с нуля.*

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
