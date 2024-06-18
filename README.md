PPO 

This is my current understanding of PPO. 

The main goal of PPO is to improve training stability of policy by limiting the change you make to a policy at each training epoch. We use PPO in order to avoid large policy updates. 
Smaller policy updates allow opitmal solution to converge while larger updates can take a long time to recover if a bad policy is resulted. By taking the ratio of the current policy compared to
the previous policy, we can clip the ratio in a range, [1-ep, 1 + ep], so that the current policy will not go beyong the old policy. The policy is updated with the new objective function, clipped 
surrogate objective function that will constrain the policy change in a small range. 

L^clip(theta) = E[min(rt(theta)At, clip(rt(theta), 1- ep, 1+ ep)At)]

rt(theta) = probability of taking(action | state) in the current policy / probability of taking(action | state) in the old policy, this replaces the log prob that we use in policy objective function
At is the advantage function 
clip(rt(theta), 1- ep, 1+ ep): clipping the ratio to ensure a not large policy update since the current policy cant be too different than the older one, two ways of doing that: TRPO which uses KL 
divergence constraints outside the objective function to constrain the policy update or implement PPO clip probability ratio directly in the objective function with its clipped surrogate objective 
function. 


The objective function is augmented with an error term on the values estimation and an entropy term to encourage sufficient exploration.
Final PP actor critic objective function E[L^clip(theta) - c1Lt^VF(theta) + c2S[pitheta](st)] c1 and c2 are hyperparameter constants


