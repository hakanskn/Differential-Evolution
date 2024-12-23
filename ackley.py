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


# def create_mutant(pop, i):
#     """Mutant vektör oluştur"""
#     idxs = [idx for idx in range(NP) if idx != i]
#     a, b, c = np.random.choice(idxs, 3, replace=False)
#     mutant = pop[a] + F * (pop[b] - pop[c])
#     return np.clip(mutant, BOUNDS[0], BOUNDS[1])

def create_mutant(pop, selected_indices):
    """
    Mutant vektör oluştur
    selected_indices: [a, b, c] indeksleri içeren liste
    """
    a, b, c = selected_indices
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
    """Çaprazlama işlemi"""
    trial = np.copy(target)
    j_rand = np.random.randint(0, D)  # Rastgele bir indis seç

    for j in range(D):
        # j_rand indisinde veya CR olasılığı ile mutant'tan al
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


def plot_iteration_details(X, Y, Z, population, target_idx, abc_indices, mutant, trial, iteration):
    """
    Her iterasyon için detaylı görselleştirme
    population: Mevcut popülasyon
    target_idx: Hedef birey indeksi
    abc_indices: Mutasyon için seçilen üç bireyin indeksleri [a, b, c]
    mutant: Mutant birey
    trial: Çaprazlama sonucu oluşan birey
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'İterasyon {iteration}, Birey {target_idx + 1}/{NP}', fontsize=16)

    # Tüm alt plotlara kontur çizgilerini ekle
    for ax in [ax1, ax2, ax3]:
        ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    # 1. Plot: Hedef birey ve mutasyon için seçilen bireyler
    ax1.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o', alpha=0.3)
    ax1.scatter(population[target_idx, 0], population[target_idx, 1],
                c='red', marker='*', s=200, label='Hedef Birey')
    ax1.scatter(population[abc_indices[0], 0], population[abc_indices[0], 1],
                c='blue', marker='s', s=100, label='a')
    ax1.scatter(population[abc_indices[1], 0], population[abc_indices[1], 1],
                c='green', marker='s', s=100, label='b')
    ax1.scatter(population[abc_indices[2], 0], population[abc_indices[2], 1],
                c='orange', marker='s', s=100, label='c')
    ax1.set_title('Hedef Birey ve Seçilen Bireyler')
    ax1.legend()

    # 2. Plot: Mutant birey oluşumu
    ax2.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o', alpha=0.3)
    ax2.scatter(population[target_idx, 0], population[target_idx, 1],
                c='red', marker='*', s=200, label='Hedef')
    ax2.scatter(mutant[0], mutant[1],
                c='purple', marker='D', s=150, label='Mutant')
    ax2.set_title('Mutant Birey Oluşumu')
    ax2.legend()

    # 3. Plot: Çaprazlama sonucu ve seçilim
    ax3.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o', alpha=0.3)

    # Hedef ve deneme bireylerinin uygunluk değerlerini hesapla
    target_fitness = ackley_function(population[target_idx])
    trial_fitness = ackley_function(trial)

    # Kazanan ve kaybeden bireyleri belirle
    winner = trial if trial_fitness < target_fitness else population[target_idx]
    winner_label = 'Deneme (Kazanan)' if trial_fitness < target_fitness else 'Hedef (Kazanan)'
    winner_color = 'lime' if trial_fitness < target_fitness else 'red'

    # Hedef ve deneme bireylerini çiz
    ax3.scatter(population[target_idx, 0], population[target_idx, 1],
                c='red', marker='*', s=200, label=f'Hedef ({target_fitness:.2f})')
    ax3.scatter(trial[0], trial[1],
                c='cyan', marker='P', s=150, label=f'Deneme ({trial_fitness:.2f})')

    # Kazanan bireyi vurgula
    ax3.scatter(winner[0], winner[1],
                facecolors='none', edgecolors=winner_color, marker='o', s=300,
                label=winner_label, linewidth=2)

    ax3.set_title('Çaprazlama ve Seçilim Sonucu')
    ax3.legend()

    plt.tight_layout()
    plt.show()


# def optimize_ackley():
#     """Ana optimizasyon fonksiyonu"""
#     # Başlangıç popülasyonu
#     population = create_population()
#     best_solution = None
#     best_fitness = float('inf')
#
#     # Kontur plot hazırlığı
#     X, Y, Z = plot_contour()
#
#     # 3D yüzey grafiği
#     plot_3d_surface(X, Y, Z)
#
#     # Optimizasyon döngüsü
#     for iteration in range(MAX_ITER):
#         # DE operatörleri
#         for i in range(NP):
#             # Mutasyon için bireylerin seçimi
#             idxs = [idx for idx in range(NP) if idx != i]
#             a, b, c = np.random.choice(idxs, 3, replace=False)
#
#             # Mutasyon
#             mutant = create_mutant(population, i)
#
#             # Çaprazlama
#             trial = crossover(population[i], mutant)
#
#             # Her 10 iterasyonda bir görselleştirme
#             if iteration % 10 == 0 and i % 5 == 0:
#                 plot_iteration_details(X, Y, Z, population, i, [a, b, c], mutant, trial, iteration)
#
#             # Seçilim
#             trial_fitness = ackley_function(trial)
#             if trial_fitness < ackley_function(population[i]):
#                 population[i] = trial
#
#                 if trial_fitness < best_fitness:
#                     best_fitness = trial_fitness
#                     best_solution = trial.copy()
#
#         print(f"İterasyon {iteration}: En iyi uygunluk = {best_fitness:.6f}")
#
#     return best_solution, best_fitness


# def plot_iteration_details(X, Y, Z, population, target_idx, abc_indices, mutant, trial, iteration):
#     """
#     Her iterasyon için detaylı görselleştirme
#     population: Mevcut popülasyon
#     target_idx: Hedef birey indeksi
#     abc_indices: Mutasyon için seçilen üç bireyin indeksleri [a, b, c]
#     mutant: Mutant birey
#     trial: Çaprazlama sonucu oluşan birey
#     """
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
#     fig.suptitle(f'İterasyon {iteration}', fontsize=16)
#
#     # Tüm alt plotlara kontur çizgilerini ekle
#     for ax in [ax1, ax2, ax3, ax4]:
#         ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.5)
#         ax.set_xlabel('x1')
#         ax.set_ylabel('x2')
#
#     # 1. Plot: Mevcut popülasyon ve hedef birey
#     ax1.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o', label='Popülasyon')
#     ax1.scatter(population[target_idx, 0], population[target_idx, 1],
#                 c='red', marker='*', s=200, label='Hedef Birey')
#     ax1.set_title('Popülasyon ve Hedef Birey')
#     ax1.legend()
#
#     # 2. Plot: Mutasyon için seçilen bireyler
#     ax2.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o')
#     ax2.scatter(population[abc_indices[0], 0], population[abc_indices[0], 1],
#                 c='blue', marker='s', s=100, label='a')
#     ax2.scatter(population[abc_indices[1], 0], population[abc_indices[1], 1],
#                 c='green', marker='s', s=100, label='b')
#     ax2.scatter(population[abc_indices[2], 0], population[abc_indices[2], 1],
#                 c='orange', marker='s', s=100, label='c')
#     ax2.set_title('Mutasyon İçin Seçilen Bireyler')
#     ax2.legend()
#
#     # 3. Plot: Mutant birey
#     ax3.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o')
#     ax3.scatter(population[target_idx, 0], population[target_idx, 1],
#                 c='red', marker='*', s=200, label='Hedef')
#     ax3.scatter(mutant[0], mutant[1],
#                 c='purple', marker='D', s=150, label='Mutant')
#     ax3.set_title('Mutant Birey')
#     ax3.legend()
#
#     # 4. Plot: Çaprazlama sonucu ve karşılaştırma
#     ax4.scatter(population[:, 0], population[:, 1], c='lightgray', marker='o')
#     ax4.scatter(population[target_idx, 0], population[target_idx, 1],
#                 c='red', marker='*', s=200, label='Hedef')
#     ax4.scatter(mutant[0], mutant[1],
#                 c='purple', marker='D', s=150, label='Mutant')
#     ax4.scatter(trial[0], trial[1],
#                 c='cyan', marker='P', s=150, label='Deneme')
#     ax4.set_title('Çaprazlama Sonucu')
#     ax4.legend()
#
#     plt.tight_layout()
#     plt.show()


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
        # DE operatörleri
        for i in range(NP):
            # Mutasyon için bireylerin seçimi
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)

            # Mutasyon
            mutant = create_mutant(population, [a, b, c])

            # Çaprazlama
            trial = crossover(population[i], mutant)

            # Her 10 iterasyonda bir görselleştirme
            if iteration % 10 == 0 and i % 5 == 0:  # i%5 ekleyerek her iterasyonda sadece bazı bireyleri göster
                plot_iteration_details(X, Y, Z, population, i, [a, b, c], mutant, trial, iteration)

            # Seçilim
            trial_fitness = ackley_function(trial)
            if trial_fitness < ackley_function(population[i]):
                population[i] = trial

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()

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