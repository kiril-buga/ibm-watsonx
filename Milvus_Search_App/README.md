Hello my friend, here some tips to get this app started:

- First, if you plan to do some major changes, please create a new branche so that we don't destroy everything at once if something goes wrong.
- Second, please clone the repository to your local machine. --> you figure out how to do this yourself
- Third, you might have realised that some Variables for Milvus connection are protected. So please create a new file called `.env`and define said variables aka. "MILVUS_PASSWORD". --> the password is just some API-key that you can create in IBM (... i think, or i hope.
- Forth: Create a virtual environement and install the requirenements AND THEN: to run the app, just write "python3 app.py" in the terminal (or however you call python on your laptop)
- Fifth, there's a sample bash script in the repository, to help you test the app. Just run the script and hopefully you get some results.
  - ps. Now there's two sample scripts, `example_compare.sh` --> to test the compare route, and `example_search.sh`to test the basic search on a specific file.
  - The two have different input variables, i was testing different stuff out, but it shouldn't matter right now as we are in the testing phase. Before we deploy it we can clean up.
- Note: the frist time you let this run, it might take some time, as the embedding model for the user query might take some time to load. But after that, the follow up requets will be very fast!
  - There a GET request included now that should be called each time the app is started... or something, so it will load the model in the background.

- Have Fun!
