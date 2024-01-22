# stable_diffusion_pytorch

I followed along a tutorial from Umar Jamil -> [link to his video](https://www.youtube.com/watch?v=ZBKpAp_6TGI)

#### Running the model
Download merges.txt and vocab.json from [here]() , as well as v1-5-pruned-emaonly.ckpt from [here]() and move the files into the 'data' folder (Note : the filename in this repo is tokenizer_vocab and not vocab, see demo.ipynb cell 2, line 13)  

Run the following:  
> pip install -r requirements.txt

Then run the cells in sd/demo.ipynb

#### Animation

I added the ability to animate the generation process, to use this please pass anim=True to the model.

#### Credits
- https://github.com/hkproj/pytorch-stable-diffusion/tree/main
- https://github.com/divamgupta/stable-diffusion-tensorflow
- https://github.com/kjsman/stable-diffusion-pytorch
- https://github.com/tinygrad/tinygrad/blob/master/examples/stable_diffusion.py








