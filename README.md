# HW 2: PyTorch Basics

The coding is worth 14 points and the free response questions are worth 4
points. There are two additional points for passing the `test_setup` case.
The deadline for this assignment is posted on Canvas.

## Academic integrity

This is an individual assignment. Your work must be your own. You may not work
with others. Do not submit other people's work as your own, and do not allow
others to submit your work as theirs. You may *talk* with other students about
the concepts covered by the homework, but you may not share code or answers
with them in any way. If you have a question about an error message or about
why a torch function returns what it does, post it on Piazza. If you need help
debugging your code, make a *private* post on Piazza or come to office hours.

We will use a combination of automated and manual methods for comparing your
code and free-response answers to that of other students. If we find
sufficiently suspicious similarities between your answers and those of another
student, you will both be reported for a suspected violation. If you're unsure
of the academic integrity policies, ask for help; we can help you avoid
breaking the rules, but we can't un-report a suspected violation.

By pushing your code to GitHub, you agree to these rules, and understand that
there may be severe consequences for violating them.

## Important instructions

Your work will be graded and aggregated using an autograder that will download
the code and free response questions from each student's repository. If you
don't follow the instructions, you run the risk of getting *zero points*. The
`test_setup` test case gives you extra credit for following these instructions 
and will make it possible to grade your work easily.

The essential instructions:
- Your code must be *pushed* to GitHub for us to grade them!  We will only
  grade the latest version of your code that was pushed to GitHub before the
  deadline.
- The same goes for the models you will save in this assignment in the `models`
  folder.
- Your NetID must be in the `netid` file; replace `NETID_GOES_HERE` with your
  netid.

## Late Work

In general, late work is not accepted. The autograder will only download work
from your repository that was pushed to GitHub before the deadline. However:
- Each student gets three late days to use across the entire quarter. If you
  want to use late days, put the number of late days you are using in the
  `LATE_DAYS` file. Currently, that file contains the number 0. If you want to
  use one late day, replace it with the number 1.
- If you have a personal emergency, please ask for help. You do not have to
  share any personal information with me, but I will ask you to get in touch
  with the dean who oversees your student services to coordinate
  accommodations.

## Get started: clone this repository

First, you need to clone this repository. If you haven't used `git` before,
check out this [tutorial](https://guides.github.com/activities/hello-world/)
or this [guide to cloning repositories](
https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Once you have `git` installed on your computer, you'll need the link to this
repository (find it on the github.com under "Clone or Download"), which might
look something like `git@github.com:cs449/hw2-username.git`. Then run:

``git clone YOUR-LINK``

As soon as you've downloaded it, go ahead and add your NetID to the `netid` file,
run `git add netid`, then `git commit -m "added netid"`, and `git push origin main`.
If you've successfully run those commands, you're almost done with the `test_setup`
test case.

## Environment setup on your computer

The following instructions are designed to help you get the necessary
requirements (in particular, Pytorch) installed on your computer. If you are
having a hard time installing Pytorch, check out [this
guide](https://pytorch.org/get-started/locally/) or try to complete the
assignment on Google Colab. If you need help, ask on Piazza.

The easiest way to install a set of Python packages that will work well
together is with a "virtual environment" using
[**miniconda**](https://docs.conda.io/en/latest/miniconda.html). Virtual
environments are a simple way to isolate all the dependencies for a particular
project, making it easy to work on multiple projects at once without them
interfering with each other (e.g. conflicting versions of libraries between
projects). To make sure your environment matches the testing environment that
we use for grading exactly, it's best to make a new environment for each
assignment in this course.

Install the latest version of [miniconda for your operating
system](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).
After installing you should be able to run `conda` from your terminal. If you
can't, you may need to run `source ~/.bash_profile` or restart your terminal.
If you for some reason cannot use miniconda, try [python's venv
module](https://docs.python.org/3/library/venv.html).  Once you have `conda`
set up, create a virtual environment by running:

``conda create -n cs449 python=3``

Note that this course uses **Python 3**. Python 2 will not work for these
assignments.  Once it's created, you can *activate* it with:

``conda activate cs449``

Here, `cs449` is the name for the environment. If you name it something else
and forget what you named it, you can call `conda env list` to list all your
conda environments. After activating it, you'll likely see that your terminal
prompt has changed to include `(cs449)`. Now, you can install the packages
necessary for this homework by going to the root directory of this repository
and running:

``pip install -r requirements.txt``

Note: you may need to follow [this guide](
https://pytorch.org/get-started/locally/) to install Pytorch locally.

Once these install, you're all set up to run the homework code locally!  If you
want to deactivate your environment, you can simply call `conda deactivate`.

## What to do for this assignment

In every function where you need to write code, there is a `raise
NotImplementeError` in the code. You will replace that line with code that
completes what the function docstring asks you to do.  The test cases will
guide you through the work you need to do and tell you how many points you've
earned. The test cases can be run from the root directory of this repository
with:

``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_save` will run all tests that begin with
`test_save`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Ask a question on Piazza, and we'll help you there.
