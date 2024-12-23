import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Algoritma parametreleri
NP = 50  # Popülasyon boyutu
D = 2  # Problem boyutu (2D Ackley)
F = 0.8  # Ölçeklendirme faktörü
CR = 0.9  # Çaprazlama oranı
MAX_ITER = 100
BOUNDS = [-5, 5]  # Çözüm uzayı sınırları


def ackley_function(x):
    """2D Ackley test fonksiyonu"""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)


def create_population():
    """Başlangıç popülasyonunu oluştur"""
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (NP, D))


def create_mutant(pop, i):
    """Mutant vektör oluştur"""
    idxs = [idx for idx in range(NP) if idx != i]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    mutant = pop[a] + F * (pop[b] - pop[c])
    return np.clip(mutant, BOUNDS[0], BOUNDS[1])


# def crossover(target, mutant):
#     """Çaprazlama işlemi"""
#     trial = np.copy(target)
#     cross_points = np.random.rand(D) < CR
#     if not np.any(cross_points):
#         cross_points[np.random.randint(0, D)] = True
#     trial[cross_points] = mutant[cross_points]
#     return trial


def crossover(target, mutant):
    """DE crossover operatörü"""
    trial = np.copy(target)
    j_rand = np.random.randint(0, D)  # Rastgele bir indis seç

    for j in range(D):
        if j == j_rand or np.random.rand() < CR:
            trial[j] = mutant[j]

    return trial


def plot_contour():
    """Ackley fonksiyonu kontur grafiğini hazırla"""
    x = np.linspace(BOUNDS[0], BOUNDS[1], 100)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ackley_function(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z


def plot_population(ax, population, iteration, title):
    """Popülasyonu kontur üzerinde göster"""
    ax.scatter(population[:, 0], population[:, 1], c='red', marker='o')
    ax.set_title(f'{title} (İterasyon {iteration})')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def plot_3d_surface(X, Y, Z):
    """3D yüzey grafiği çiz"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    plt.colorbar(surf)
    plt.title('Ackley Fonksiyonu 3D Görünüm')
    plt.show()


def optimize_ackley():
    """Ana optimizasyon fonksiyonu"""
    # Başlangıç popülasyonu
    population = create_population()
    best_solution = None
    best_fitness = float('inf')

    # Kontur plot hazırlığı
    X, Y, Z = plot_contour()

    # 3D yüzey grafiği
    plot_3d_surface(X, Y, Z)

    # Optimizasyon döngüsü
    for iteration in range(MAX_ITER):
        # Görselleştirme (her 10 iterasyonda bir)
        if iteration % 10 == 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Kontur çizimi
            for ax in [ax1, ax2, ax3]:
                ax.contour(X, Y, Z, levels=20)

            # Mevcut popülasyon
            plot_population(ax1, population, iteration, "Popülasyon")

        # DE operatörleri
        for i in range(NP):
            # Mutasyon
            mutant = create_mutant(population, i)

            if iteration % 10 == 0:
                # Mutant vektörleri göster
                temp_pop = np.vstack([population, mutant])
                plot_population(ax2, temp_pop, iteration, "Mutasyon")

            # Çaprazlama
            trial = crossover(population[i], mutant)

            # Seçilim
            trial_fitness = ackley_function(trial)
            if trial_fitness < ackley_function(population[i]):
                population[i] = trial

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()

        if iteration % 10 == 0:
            # Son popülasyonu göster
            plot_population(ax3, population, iteration, "Seçilim")
            plt.tight_layout()
            plt.show()

        print(f"İterasyon {iteration}: En iyi uygunluk = {best_fitness:.6f}")

    return best_solution, best_fitness


if __name__ == "__main__":
    # Rastgele sayı üretecini sabitleyerek tekrarlanabilirlik sağla
    np.random.seed(42)

    # Optimizasyonu çalıştır
    best_solution, best_fitness = optimize_ackley()

    print("\nOptimizasyon tamamlandı!")
    print(f"En iyi çözüm: {best_solution}")
    print(f"En iyi uygunluk değeri: {best_fitness}")