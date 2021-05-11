# CityBrainChallenge-starter-kit

### FAQ

1. 
```
IndexError: basic_string::at: __n (which is 18446744073709551615) >= this->size() (which is 2)
```

If your operating system is Windows and you have set `git config --global core.autocrlf true` , when you clone this repo, git will automatically add CR to cfg/simulator.cfg. This will lead to the error in Linux of the docker container.

So please change cfg/simulator.cfg from CRLF to LF after cloning this repo.
