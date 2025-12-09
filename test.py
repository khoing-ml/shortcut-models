print('Starting smoke test: encode 16 samples via get_dataset + StableVAE')
try:
    from encode_and_cluster import encode_with_stablevae_from_iterator
    from utils.datasets import get_dataset
    it = get_dataset('celebahq256', 4, True, False)
    latents, paths = encode_with_stablevae_from_iterator(it, n_samples=16)
    print('SUCCESS: latents shape =', latents.shape)
    print('num paths =', len(paths))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('FAILED:', e)