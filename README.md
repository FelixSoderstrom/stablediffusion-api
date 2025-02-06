# Important
**We need to run 3.10 here!**
There is an issue with torch and 3.13.
3.10 was the most stable version for stable diffusion.

We ran the entire (today backend) project on 3.10 because of this.

Since this repo will just be an API, we can safely use separate python version across the backend of this whole application.

If you try to run this code on python 3.13 with the current structure of the image generation code we have you will get import errors because of the structural change in pydantic.
Dont know if pydantic is needed here.
But lets just keep to 3.10 to be safe. 
Happy days, catch you later.

Frivilliga styrkekramar
Undertecknad
10x