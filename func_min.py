import numpy as np

# Функція, яку будемо оптимізувати (мінімізувати)
def objective_function(x):
    return (x[1] - 3) * np.exp((-x[0] ** 2 - x[1] ** 2))

# Налаштування генетичного алгоритму
POPULATION_SIZE = 100  # Розмір популяції
NUM_GENERATIONS = 100  # Кількість поколінь
MUTATION_RATE = 0.1  # Ймовірність мутації
CROSSOVER_RATE = 0.8  # Ймовірність кросинговеру
BOUNDS = [(-1000, 1000), (-1000, 1000)]  # Межі для кожної змінної

# Ініціалізація популяції
def initialize_population(size, bounds):
    return np.array([np.random.uniform(b[0], b[1], size) for b in bounds]).T

# Оцінка пристосованості (fitness) для кожного індивіда
def evaluate_fitness(population):
    return np.array([objective_function(ind) for ind in population])

# Турнірний відбір
def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(range(len(population)), k, replace=False)
        best = indices[np.argmin(fitness[indices])]
        selected.append(population[best])
    return np.array(selected)

# Кросинговер (одноточковий)
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    return parent1, parent2

# Мутація (гаусівська)
def mutate(individual, bounds):
    if np.random.rand() < MUTATION_RATE:
        mutation = np.random.normal(0, 0.1, size=len(individual))  # Маленькі стрибки для всіх змінних
        individual += mutation
        for i in range(len(individual)):
            individual[i] = np.clip(individual[i], bounds[i][0], bounds[i][1])  # Залишаємо в межах
    return individual

# Основний цикл генетичного алгоритму
population = initialize_population(POPULATION_SIZE, BOUNDS)
for generation in range(NUM_GENERATIONS):
    fitness = evaluate_fitness(population)
    selected = tournament_selection(population, fitness)

    # Створення нового покоління
    next_generation = []
    for i in range(0, len(selected), 2):
        parent1, parent2 = selected[i], selected[min(i + 1, len(selected) - 1)]
        child1, child2 = crossover(parent1, parent2)
        next_generation.append(mutate(child1, BOUNDS))
        next_generation.append(mutate(child2, BOUNDS))

    population = np.array(next_generation[:POPULATION_SIZE])  # Оновлення популяції

    # Вивід результатів для кожного покоління
    best_fitness = np.min(fitness)
    best_individual = population[np.argmin(fitness)]
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.10f}, Best Individual = {best_individual}")

# Результати
final_fitness = evaluate_fitness(population)
best_fitness = np.min(final_fitness)
best_individual = population[np.argmin(final_fitness)]
print("\nFinal Solution:")
print(f"Best Fitness = {best_fitness:.10f}, Best Individual = {best_individual}")
