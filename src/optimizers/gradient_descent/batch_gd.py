# src/optimizers/gradient_descent/batch_gd.py


class BatchGD:
    """Батч-градиентный спуск: шаг против градиента."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate  # Длина шага

    def step(self, params, grads):  # step (шаг) - одни шаг, обновление
        return [p - self.learning_rate * g  # новый параметр = старый - шаг * градиент
                for p, g in zip(params, grads)] # zip (застежка) - идем парами: веса и его градиент


if __name__ == "__main__":

    opt = BatchGD(learning_rate=0.1)
    x = 0.0     # стартуем далеко от минимума

    for _ in range(100):
        grad = 2 * (x - 3)      # градиент в текущей точке
        [x] = opt.step([x], [grad])     # шаг: результат - список из одного элемента
    print(x)    # ~3.0 - нашел минимум