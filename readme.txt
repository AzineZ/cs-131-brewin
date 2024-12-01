Right now, the interpreter is failing Correctness | test_lazy_mutation3 (0/1). It's also failing a shadowing issue when in a function foo,
declaring variable named a with a parameter named leads to the variable defined error to be incorrectly raised. Not sure if the test case 
relates to the shadowing error. 