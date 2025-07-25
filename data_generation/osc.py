# trial

from torch.utils.data import Dataset, DataLoader
import torch
import random
import itertools
import numpy as np
from scipy.integrate import solve_ivp

class Dampened_Oscillator(Dataset):

    '''
    Generates a dampened oscillator dataset and returns it.
    '''

    def __init__(self, cfg_dampo):
        super().__init__()

        self.m_list = cfg_dampo['m_list']
        self.c_list = cfg_dampo['c_list']
        self.k_list = cfg_dampo['k_list']
        self.system_type = cfg_dampo['system_type']

        self.x0_list = cfg_dampo['x0_list']
        self.v0_list = cfg_dampo['v0_list']
        self.initial_type = cfg_dampo['initial_type']

        self.t_list = cfg_dampo['t_list']

        if self.system_type == 'basiccomb': # Randomly pick combinations

            assert cfg_dampo['system_n'] is not None, 'Specify sample amount for this generation type system coeffs.'

            self.system_n = cfg_dampo['system_n']

            # Create combinations 

            self.system_coeff_matrix = []

            for _ in range(self.system_n):

                row = [
                    random.choice(self.m_list),
                    random.choice(self.c_list),
                    random.choice(self.k_list)
                ]

                (self.system_coeff_matrix).append(row)

            self.system_coeff_matrix = torch.tensor(self.system_coeff_matrix)

        elif self.system_type == 'fullcomb': # Return all the possible combinations

            self.system_coeff_matrix = list(itertools.product(self.m_list, self.c_list, self.k_list))
            random.shuffle(self.system_coeff_matrix)
            self.system_coeff_matrix = torch.tensor(self.system_coeff_matrix, dtype=torch.float32)

        else: 
            print(f'Unknown system combination type: {self.system_type}')
        
        if self.initial_type == 'basiccomb': # Randomly pick combinations

            assert cfg_dampo['initial_n'] is not None, 'Specify sample amount for this generation type for init conds.'

            self.initial_n = cfg_dampo['initial_n']

            # Create combinations 

            self.initial_cond_matrix = []

            for _ in range(self.initial_n):

                row = [
                    random.choice(self.x0_list),
                    random.choice(self.v0_list)
                ]

                (self.initial_cond_matrix).append(row)

            self.initial_cond_matrix = torch.tensor(self.initial_cond_matrix)

        elif self.initial_type == 'fullcomb': # Return all the possible combinations

            self.initial_cond_matrix = list(itertools.product(self.x0_list, self.v0_list))
            random.shuffle(self.initial_cond_matrix)
            self.initial_cond_matrix = torch.tensor(self.initial_cond_matrix, dtype=torch.float32)

        else: 
            print(f'Unknown system combination type: {self.initial_type}')
        
        # Now that we have the matrices with combinations generated, we will solve the systems. And find the x values for each cell in t_list

        self.solutions, self.combined_cond = Dampened_Oscillator._solve_oscillator(self.system_coeff_matrix, self.initial_cond_matrix, self.t_list)

        del self.system_coeff_matrix, self.initial_cond_matrix # Clear memory

    @staticmethod
    def _solve_oscillator_rhs(t, y, m, c, k):
        x, v = y
        dxdt = v
        dvdt = -(c/m)*v -(k/m)*x
        return [dxdt, dvdt]

    @staticmethod
    def _solve_oscillator(system_coeff_matrix: torch.tensor, init_cond_matrix: torch.tensor, t_list: torch.tensor):

        """

        Parameters:
        System coeff matrix of n samples and 3 coeffs: ()
        """

        solutions = np.zeros(shape=(system_coeff_matrix.shape[0] * init_cond_matrix.shape[0], len(t_list)))
        combined_params = np.zeros(shape=(system_coeff_matrix.shape[0] * init_cond_matrix.shape[0], 5))

        idx = 0
        for i in range(system_coeff_matrix.shape[0]):
            for j in range(init_cond_matrix.shape[0]):

                m, c, k = system_coeff_matrix[i]
                x0, v0 = init_cond_matrix[j]

                sol = solve_ivp(
                    Dampened_Oscillator._solve_oscillator_rhs,
                    (t_list[0], t_list[-1]),
                    y0= [x0, v0],
                    t_eval= t_list,
                    args= (m,c,k),
                    method= 'RK45'
                    )

                solutions[idx] = sol.y[0]
                combined_params[idx] = [x0, v0, m, c, k]
                
                idx = idx+1
            
        return torch.tensor(solutions, dtype=torch.float32), torch.tensor(combined_params, dtype=torch.float32)

    def __len__(self):
        return self.solutions.shape[0]

    def __getitem__(self, idx):
        return self.solutions[idx], self.combined_cond[idx]


