Notes:

I ran the toy_grid_day.py file to train the GFN and MCMC samplers. The specific commands to train each sampler are as follows:
- MCMC: 
    python3 toy_grid_dag.py --func='cos_N' --method='mcmc' --save_path='results/toy_mcmc_cosN.pkl.gz'
- GFN: 
    python3 toy_grid_dag.py --func='cos_N' --method='flownet' --save_path='results/toy_gfn_cosN.pkl.gz'

Once you get results, unzip the .gz files and then you can plot the results using plot_paths.py. For my results, I used the following arguments:
- python3 plot_paths.py --filename='toy_gfn_cosN'
- python3 plot_paths.py --filename='toy_gfn_cosN' --full_path=False
- python3 plot_paths.py --filename='toy_mcmc_cosN'

Note: the "example_branincurrin.pkl" and "flow_insp_0.pkl" files in the results/ directory are other trained GFNs. However, I have not yet been able to fully figure out how to properly sample and plot the results.