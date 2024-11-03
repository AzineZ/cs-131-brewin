So far, I still have test_ret3 failing on me. Below are my biggest issues that I HAVE solved:
- isinstance convert True to integer. 
- Scoping: backtracking didn't stop at function caller level.
- env stack handling.
- 