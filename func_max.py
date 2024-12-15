import numpy as np

# Функція, яку будемо оптимізувати (максимізувати)
def objective_function(x):
    return 3 * np.cbrt((x + 4) ** 2) - 2 * 2 - 8

# Налаштування генетичного алгоритму
POPULATION_SIZE = 100  # Розмір популяції
NUM_GENERATIONS = 100  # Кількість поколінь
MUTATION_RATE = 0.1  # Ймовірність мутації
CROSSOVER_RATE = 0.8  # Ймовірність кросинговеру
BOUNDS = (-1000, 1000)  # Межі для x

# Ініціалізація популяції
def initialize_population(size, bounds):
    return np.random.uniform(bounds[0], bounds[1], size)

# Оцінка пристосованості (fitness) для кожного індивіда
def evaluate_fitness(population):
    return np.array([objective_function(x) for x in population])

# Турнірний відбір
def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(range(len(population)), k, replace=False)
        best = indices[np.argmax(fitness[indices])]
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
        mutation = np.random.normal(0, 0.1)  # Маленький стрибок
        individual += mutation
        individual = np.clip(individual, bounds[0], bounds[1])  # Залишаємо в межах
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
    best_fitness = np.max(fitness)
    best_individual = population[np.argmax(fitness)]
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}, Best Individual = {best_individual:.4f}")

# Результати
final_fitness = evaluate_fitness(population)
best_fitness = np.max(final_fitness)
best_individual = population[np.argmax(final_fitness)]
print("\nFinal Solution:")
print(f"Best Fitness = {best_fitness:.4f}, Best Individual = {best_individual:.4f}")