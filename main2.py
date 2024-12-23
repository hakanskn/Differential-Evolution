"""
Rüzgâr Türbini Tasarımı Optimizasyonu Senaryosu (5 Boyutlu)

Parametreler:
1. Kanat Uzunluğu (L): [10, 50] metre
2. Eksenel Hız Katsayısı (A): [0.5, 1.5]
3. Kanat Kalınlığı (B): [0.1, 1.0] metre
4. Kanat Eğimi (T): [0°, 20°]
5. Kanat Sayısı (N): [2, 5]

Amaç:
Belirli bir hedef güce (200 kW) mümkün olduğunca yakın olacak şekilde
toplam maliyeti minimize etmek.

Maliyet Fonksiyonu:
Maliyet_toplam = Maliyet_üretim + Maliyet_bakım + Ceza

Maliyet_üretim = (L^2 * 800) + (B^2 * 600) + (N * 200)
Maliyet_bakım = (A^2 * 500) + (T * 50)
G(L,A,B,T,N) = 4 * L * A * (B+0.5) * (N/3) * cos(T rad)
Ceza = |G(L,A,B,T,N) - 200| * 50

DE Parametreleri:
- NP (Popülasyon Boyutu): 20
- D (Boyut sayısı): 5
- Maks. İterasyon: 100
- F: 0.8 (Ölçeklendirme Faktörü)
- CR: 0.9 (Crossover Oranı)
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Sabitler
NP = 20  # Popülasyon boyutu
D = 5  # Boyut sayısı (L,A,B,T,N)
F = 0.8  # Ölçeklendirme faktörü
CR = 0.9  # Crossover oranı
MAX_ITER = 100
TARGET_POWER = 200  # Hedef güç (kW)

# Parametre sınırları [min, max]
BOUNDS = np.array([
    [10.0, 50.0],  # L: Kanat Uzunluğu (m)
    [0.5, 1.5],  # A: Eksenel Hız Katsayısı
    [0.1, 1.0],  # B: Kanat Kalınlığı (m)
    [0.0, 20.0],  # T: Kanat Eğimi (derece)
    [2.0, 5.0]  # N: Kanat Sayısı
])


def calculate_power(params):
    """Türbin güç hesaplama fonksiyonu"""
    L, A, B, T, N = params
    rad = np.radians(T)
    return 4 * L * A * (B + 0.5) * (N / 3.0) * np.cos(rad)


def calculate_costs(params):
    """Maliyet bileşenlerini hesaplama fonksiyonu"""
    L, A, B, T, N = params

    production_cost = (L ** 2 * 800) + (B ** 2 * 600) + (N * 200)
    maintenance_cost = (A ** 2 * 500) + (T * 50)
    power = calculate_power(params)
    penalty = abs(power - TARGET_POWER) * 50

    return production_cost, maintenance_cost, penalty


def fitness_function(individual):
    """Amaç fonksiyonu"""
    prod_cost, maint_cost, penalty = calculate_costs(individual)
    return prod_cost + maint_cost + penalty


def initialize_population():
    """Başlangıç popülasyonunu oluştur"""
    pop = np.zeros((NP, D))
    for i in range(D):
        pop[:, i] = np.random.uniform(BOUNDS[i, 0], BOUNDS[i, 1], NP)
    return pop


def mutation(pop):
    """DE mutasyon operatörü"""
    mutant = np.zeros_like(pop)
    for i in range(NP):
        # 3 rastgele indeks seç
        idxs = [idx for idx in range(NP) if idx != i]
        a, b, c = np.random.choice(idxs, 3, replace=False)

        # Mutant vektör
        mutant_vector = pop[a] + F * (pop[b] - pop[c])

        # Sınırları kontrol et
        for j in range(D):
            mutant_vector[j] = np.clip(mutant_vector[j], BOUNDS[j, 0], BOUNDS[j, 1])

        mutant[i] = mutant_vector
    return mutant


# def crossover(pop, mutant):
#     """DE crossover operatörü"""
#     trial = np.zeros_like(pop)
#     for i in range(NP):
#         cross_points = np.random.rand(D) < CR
#         if not np.any(cross_points):
#             cross_points[np.random.randint(0, D)] = True
#         trial[i] = np.where(cross_points, mutant[i], pop[i])
#     return trial


def crossover(pop, mutant):
    """DE crossover operatörü"""
    trial = np.zeros_like(pop)

    for i in range(NP):
        trial[i] = pop[i].copy()
        j_rand = np.random.randint(D)  # Her birey için rastgele bir indis

        for j in range(D):
            if j == j_rand or np.random.rand() < CR:
                trial[i, j] = mutant[i, j]

    return trial

def selection(pop, trial):
    """DE seçim operatörü"""
    for i in range(NP):
        if fitness_function(trial[i]) < fitness_function(pop[i]):
            pop[i] = trial[i]
    return pop


def plot_convergence(costs_history):
    """Yakınsama grafiği"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs_history, 'b-', label='En İyi Maliyet')
    plt.title('Optimizasyon Yakınsama Grafiği')
    plt.xlabel('İterasyon')
    plt.ylabel('Toplam Maliyet')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_turbine_params(best_solutions):
    """Parametre değişim grafiği"""
    param_names = ['Kanat Uzunluğu (m)', 'Eksenel Hız Katsayısı',
                   'Kanat Kalınlığı (m)', 'Kanat Eğimi (°)', 'Kanat Sayısı']

    plt.figure(figsize=(12, 8))
    for i in range(D):
        plt.subplot(D, 1, i + 1)
        plt.plot(best_solutions[:, i])
        plt.ylabel(param_names[i])
        plt.grid(True)
    plt.xlabel('İterasyon')
    plt.tight_layout()
    plt.show()


def plot_cost_components(best_solution):
    """Maliyet bileşenleri pasta grafiği"""
    prod_cost, maint_cost, penalty = calculate_costs(best_solution)
    costs = [prod_cost, maint_cost, penalty]
    labels = ['Üretim Maliyeti', 'Bakım Maliyeti', 'Güç Sapması Cezası']

    plt.figure(figsize=(8, 8))
    plt.pie(costs, labels=labels, autopct='%1.1f%%')
    plt.title('Maliyet Bileşenleri Dağılımı')
    plt.show()


def visualize_turbine(params):
    """Basit türbin görselleştirmesi"""
    L, _, B, T, N = params

    # Kanat açıları
    angles = np.linspace(0, 360, int(N), endpoint=False)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')

    # Kanatları çiz
    for angle in angles:
        rad_angle = np.radians(angle)
        # Kanat şeklini çiz
        blade_x = np.linspace(0, L, 100)
        blade_y = np.zeros_like(blade_x)
        blade_y += B * np.sin(np.pi * blade_x / L)  # Kanat eğimi

        # Koordinat dönüşümü
        rotated_x = blade_x * np.cos(rad_angle) - blade_y * np.sin(rad_angle)
        rotated_y = blade_x * np.sin(rad_angle) + blade_y * np.cos(rad_angle)

        ax.plot(np.arctan2(rotated_y, rotated_x), np.sqrt(rotated_x ** 2 + rotated_y ** 2), 'b-')

    ax.set_title(f'Türbin Üstten Görünüm\nKanat Sayısı: {N}, Kanat Uzunluğu: {L:.1f}m')
    plt.show()


def optimize_turbine():
    """Ana optimizasyon fonksiyonu"""
    # Başlangıç popülasyonu
    pop = initialize_population()

    # Tarihçe
    cost_history = []
    solution_history = []
    best_solution = None
    best_cost = float('inf')

    # Ana döngü
    for iteration in range(MAX_ITER):
        # DE operatörleri
        mutant = mutation(pop)
        trial = crossover(pop, mutant)
        pop = selection(pop, trial)

        # En iyi çözümü güncelle
        current_best_idx = min(range(NP), key=lambda i: fitness_function(pop[i]))
        current_cost = fitness_function(pop[current_best_idx])

        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = pop[current_best_idx].copy()

        cost_history.append(best_cost)
        solution_history.append(best_solution.copy())

        if iteration % 10 == 0:
            print(f"İterasyon {iteration}: En iyi maliyet = {best_cost:.2f}")

    # Sonuçları görselleştir
    solution_history = np.array(solution_history)

    print("\nOptimizasyon Tamamlandı!")
    print(f"En İyi Çözüm:")
    print(f"Kanat Uzunluğu: {best_solution[0]:.2f} m")
    print(f"Eksenel Hız Katsayısı: {best_solution[1]:.2f}")
    print(f"Kanat Kalınlığı: {best_solution[2]:.2f} m")
    print(f"Kanat Eğimi: {best_solution[3]:.2f}°")
    print(f"Kanat Sayısı: {int(round(best_solution[4]))}")
    print(f"Toplam Maliyet: {best_cost:.2f}")
    print(f"Üretilen Güç: {calculate_power(best_solution):.2f} kW")

    # Görselleştirmeler
    plot_convergence(cost_history)
    plot_turbine_params(solution_history)
    plot_cost_components(best_solution)
    visualize_turbine(best_solution)

    return best_solution, best_cost


if __name__ == "__main__":
    best_solution, best_cost = optimize_turbine()