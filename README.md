以下是 assignment 4-2 的source code

ℹ️
arl_results.csv 是數據


```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def calculate_control_limits(mu0, sigma0, n, alpha=0.0027):
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)
    UCL = mu0 + z_alpha_2 * (sigma0 / np.sqrt(n))
    LCL = mu0 - z_alpha_2 * (sigma0 / np.sqrt(n))  #這應該無需多言
    return UCL, LCL

def simulate_arl(mu, sigma0, n, UCL, LCL, max_samples=10000):
    for i in range(1, max_samples + 1):    #ith sampling
        sample = np.random.normal(mu, sigma0, n)  #用normal ，mu : 可能是mu0 or mu1 
        xbar = np.mean(sample) #樣本 mean
        
        if xbar > UCL or xbar < LCL: #超出管制界線
            return i#回報第幾次
    
    return max_samples

def calculate_theoretical_arl(mu0, mu1, sigma0, n, alpha=0.0027):
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)
    
    if mu1 == mu0:
        return 1 / alpha
    
    delta = abs(mu1 - mu0) / (sigma0 / np.sqrt(n))
    
    if mu1 > mu0:
        beta = stats.norm.cdf(z_alpha_2 - delta) - stats.norm.cdf(-z_alpha_2 - delta)
    else:
        beta = stats.norm.cdf(z_alpha_2 + delta) - stats.norm.cdf(-z_alpha_2 + delta) #簡報有說明
    
    return 1 / (1 - beta)

def run_simulation(mu0=1.5, sigma0=0.15, n_values=[3, 5, 10, 15, 20], 
                   k_values=[0, 0.5, 1.0, 1.5, 2.0, 2.5], replications=10000):
    
    results = []

    print(f"mu0 = {mu0}, sigma0 = {sigma0}")
    print(f"Cycles = {replications}\n")
    
    for n in n_values:
        print(f"Processing n = {n}")
        
        UCL, LCL = calculate_control_limits(mu0, sigma0, n)
        print(f"  UCL = {UCL:.4f}, LCL = {LCL:.4f}")
        
        for k in k_values:
            mu1 = mu0 + k * sigma0 #μ₁ = Mu₀ + k × Sigma₀
            
            arl_theoretical = calculate_theoretical_arl(mu0, mu1, sigma0, n)#算ARL
            
            np.random.seed(42)#隨機SEED，改了數據會變
            arl_simulated_list = []
            for rep in range(replications):
                arl = simulate_arl(mu1, sigma0, n, UCL, LCL) #每次RETURN run length 
                arl_simulated_list.append(arl)#存
            
            arl_simulated = np.mean(arl_simulated_list) 
            
            results.append({
                'n': n,
                'k': k,
                'mu1': mu1,
                'ARL_theoretical': arl_theoretical,
                'ARL_simulated': arl_simulated,
                'difference': abs(arl_theoretical - arl_simulated),
                'relative_error': abs(arl_theoretical - arl_simulated) / arl_theoretical * 100
            })
            
            print(f"  k = {k}: ARL_theo = {arl_theoretical:.2f}, ARL_sim = {arl_simulated:.2f}")
    
    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    mu0 = 1.5
    sigma0 = 0.15
    n_values = [3, 5, 10, 15, 20]
    k_values = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    replications = 10000

    df_results = run_simulation(mu0, sigma0, n_values, k_values, replications)

    df_results.to_csv('arl_results.csv', index=False)
```
