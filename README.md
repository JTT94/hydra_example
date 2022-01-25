# hydra_example

## Run command line
-----------------

- Overide defaults on command line with extra flafs
- Hierarchical config so 'num_epochs' in 'training' sub-config may also be overridden
`
  python main.py experiment=vae_example training.num_epochs=50
`

## Load config interactively in notebook
-----------------

```
    import hydra
    from hydra import compose, initialize
    fdir = get_directory(config_filepath)
    fdir = os.path.relpath(fdir, start = os.curdir)
    
    fname = get_filename(config_filepath, extension=False)
    
    clear_hydra()
    initialize(config_path=fdir)
    cfg = compose(config_name=fname)
```

## Init classes from config
-----------------

- model yaml corresponding to cfg.model

```
  _target_: src.vae.VAE # class location
  input_dim: 784
  hidden_dim1: 400
  hidden_dim2: 20
```

- run.py

```
  from hydra.utils import instantiate, get_class, call
  model = instantiate(cfg.model).to(device)
```
