Notes:

# 4/28/22
I was still having trouble figuring out the issue with the EBengio code this week so I used tikquuss's implementation instead. I was able to achieve the expected results with the GFN sampler. Raw sample results are in the results/ directory and plots of the reward functions and samples are in the figures directory.

I defined the following loss function to compare different samplers (not sure if this is already a thing). Since the exact probability distribution of the cosine function is known, I calculated the loss of a sample draw by summing the squared difference of the Monte Carlo distrbutions (i.e., sum of draws over total draws) from the true probabilities. This loss evaluator is in the file plot_loss.py

Note: some of the pkl files are too large to push to git.


# 4/21/22
I ran the eb_gfn_grid.py file to train the GFN and MCMC samplers. The specific commands to train each sampler are as follows:
- MCMC: 
    python3 eb_gfn_grid.py --func='cos_N' --method='mcmc' --save_path='results/toy_mcmc_cosN.pkl.gz'
- GFN: 
    python3 eb_gfn_grid.py --func='cos_N' --method='flownet' --save_path='results/toy_gfn_cosN.pkl.gz'

Once you get results, unzip the .gz files and then you can plot the results using plot_paths.py. For my results, I used the following arguments:
- python3 plot_paths.py --filename='toy_gfn_cosN'
- python3 plot_paths.py --filename='toy_gfn_cosN' --full_path=False
- python3 plot_paths.py --filename='toy_mcmc_cosN'

Note: the "example_branincurrin.pkl" and "flow_insp_0.pkl" files in the results/ directory are other trained GFNs. However, I have not yet been able to fully figure out how to properly sample and plot the results.