import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from homework4 import Hw5Env, CNP, bezier
from tqdm import tqdm


def collect_trajectories(num_dms = 500,steps= 100, render_mode= "offscreen"):
    env = Hw5Env(render_mode=render_mode)
    states_arr = []

    for i in range(num_dms):
        env.reset()

        p1 = np.array([0.5,  0.30, 1.04])
        p2 = np.array([0.5,  0.15, np.random.uniform(1.04, 1.40)])
        p3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.40)])
        p4 = np.array([0.5, -0.30, 1.04])
        curve = bezier(np.stack([p1, p2, p3, p4], axis=0), steps)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180],
                                 n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())         # (e_y, e_z, o_y, o_z, h)
        states_arr.append(np.stack(states))
        print(f"Collected {i+1} trajectories.", end="\r")

    np.save("trajectories.npy",np.array(states_arr,dtype=object))
    print(f"[data] Collected {num_dms} trajectories.")
    return np.array(states_arr,dtype=object)

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    num_samples = data.shape[0]
    t = np.linspace(0, 1, data.shape[1]).reshape(-1, 1)
    t = np.tile(t, (data.shape[0], 1, 1))
    data = np.concatenate((data, t), axis=2) # (e_y, e_z, o_y, o_z, h, t)


    train_samples = int(num_samples * train_ratio)
    val_samples = int(num_samples * val_ratio)
    test_samples = num_samples - train_samples - val_samples
    train_data = data[:train_samples]
    val_data = data[train_samples:train_samples + val_samples]
    test_data = data[train_samples + val_samples:]
    
    np.save("train_data.npy", train_data)
    np.save("val_data.npy", val_data)
    np.save("test_data.npy", test_data)
    print(f"[data] Split dataset into train, val, and test sets.")
    return train_data, val_data, test_data

    
def sample_batch(data,batch_size,max_targets=10):
    
    T = data.shape[1]
    obs = torch.zeros((batch_size, 10, 6))
    obs_mask = torch.zeros((batch_size, 10))
    target_query = torch.zeros((batch_size, max_targets, 2))
    target_truth = torch.zeros((batch_size, max_targets, 4))
    target_mask = torch.zeros((batch_size, max_targets))
 
    idxs = np.random.randint(0,len(data),batch_size)
    
    for i, j in enumerate(idxs):
        seq = data[j]        

        n_c = np.random.randint(1, 11)
        n_t = np.random.randint(1, max_targets + 1)
        
        indices = np.random.choice(T, n_c + n_t, replace=False)
        c_ids   = indices[:n_c]
        t_ids   = indices[n_c:]
        
        for ci, idx in enumerate(c_ids):
            e_y, e_z, o_y, o_z, h, t = seq[idx]
            obs[i, ci]      = torch.tensor([t, h, e_y, e_z, o_y, o_z], dtype=torch.float32)
            obs_mask[i, ci] = 1.0
        
        for ti, idx in enumerate(t_ids):
            e_y, e_z, o_y, o_z, h, t = seq[idx]
            target_query[i, ti] = torch.tensor([t, h], dtype=torch.float32)
            target_truth[i, ti] = torch.tensor([e_y, e_z, o_y, o_z], dtype=torch.float32)
            target_mask[i, ti] = 1.0

    return obs, target_query, target_truth, obs_mask, target_mask


def train(data, epochs, batch_size, max_targets):
    
    train_data, val_data, _ = split_dataset(data)
    model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 32

    train_losses = []
    min_val_loss = np.inf
    for ep in tqdm(range(1, epochs+1)):
        observation, target_q, target_truth, obs_mask, target_mask = sample_batch(train_data, batch_size, max_targets)

        loss = model.nll_loss(observation, target_q, target_truth, observation_mask=obs_mask, target_mask=target_mask)
        train_losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if ep % 200 == 0:
            with torch.no_grad():
                val_loss = 0
                for i in range(val_data.shape[0]):
                    observation, target_q, target_truth, obs_mask, target_mask = sample_batch(val_data, batch_size, max_targets)
                    val_loss += model.nll_loss(observation, target_q, target_truth, observation_mask=obs_mask, target_mask=target_mask).item()
                val_loss /= val_data.shape[0]
            
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"model_best.pt")
                tqdm.write(f"[train] epoch {ep:3d}/{epochs}  NLL = {loss.item():.4f}  val_loss = {val_loss:.4f}")
    np.save("train_losses.npy", np.array(train_losses))
    

def plot_train_loss(train_losses):
    plt.figure(figsize =(8,8))
    plt.plot(train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("NLL Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("train_loss.png", dpi=200)
    plt.close()
    print("Saved training loss plot as train_loss.png")

def bar_plot(mse_obj, mse_ee, fname="results.png"):
    means = [mse_obj.mean(), mse_ee.mean()]
    stds  = [mse_obj.std(),  mse_ee.std()]

    plt.figure(figsize=(4,4))
    plt.bar(['Object', 'End‑effector'], means, yerr=stds, capsize=10)
    plt.ylabel("Mean Squared Error")
    plt.title("CNMP ‑ 100 random tests")
    plt.tight_layout()
    plt.ylim(0, 0.001)
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved bar plot of errors as", fname)

def test(num_tests = 100, batch_size = 1, max_targets = 1):

    test_data = np.load("test_data.npy", allow_pickle=True) 
    
    model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3)
    model.load_state_dict(torch.load("model_best.pt"))
    model.eval()

    mse_obj_list = []
    mse_ee_list  = []

    for _ in range(num_tests):

        obs, target_query, target_truth, obs_mask, target_mask = sample_batch(test_data, batch_size = 1, max_targets=max_targets)

        with torch.no_grad():
            mu, _ = model(obs, target_query, observation_mask=obs_mask)
        
  
        mu_np    = mu.cpu().numpy().reshape(-1, 4)
        truth_np = target_truth.cpu().numpy().reshape(-1, 4)

        mse_obj = np.mean((mu_np[:, 2:] - truth_np[:, 2:]) ** 2)
        mse_ee  = np.mean((mu_np[:, :2] - truth_np[:, :2]) ** 2)
        mse_obj_list.append(mse_obj)
        mse_ee_list.append(mse_ee)
    

    mse_ee_arr  = np.array(mse_ee_list)
    mse_obj_arr = np.array(mse_obj_list)
    np.save("mse_ee.npy", mse_ee_arr)
    np.save("mse_obj.npy", mse_obj_arr)

    bar_plot(mse_obj_arr, mse_ee_arr, fname="results.png")
  

if __name__ == "__main__":
    # collect_trajectories(num_dms=500, steps=100, render_mode="offscreen")
    #data = np.load("trajectories.npy", allow_pickle=True)
    #train(data, epochs=50000, batch_size=32, max_targets=1)
    plot_train_loss(np.load("train_losses.npy"))
    test(num_tests=100, batch_size=1, max_targets=1)
   

    