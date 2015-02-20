# CUDABiconnectedComponents
An implementation of a biconnected components finding algorithm for CUDA


Przydatne info:

Każda funkcja z klasy BiconnectedComponents powinna być zaimplementowana w osobnym pliku.

Schemat: (do napisania funkcja foo)

1) tworzymy pliki foo.cu i fooMockup.cu w folderze src/

fooMockup.cu ma wyglądać tak:

    #include "config.h"
    #ifndef foo_IMPLEMENTED

    // prosta implementacja

    #endif
    
foo.cu tak:
    
    #include "config.h"
    #ifdef foo_IMPLEMENTED

    // równoległa implementacja

    #endif

2) Na końcu należy odkomentować odpowiednią linię w pliku config.h

    #define foo_IMPLEMENTED



Testowanie:

Makefile nie tworzy pliku wykonywalnego.
Takowe powstają dopiero przy "make test" - ta funkcjonalność działa w sposób następujący:

1) kompiluje plik main.o w katalogu test

2) linkuje go z implementacją klasy

3) uruchamia binarkę

Wobec tego, polecenie "make main.o" w katalogu test/ powinno kompilować maina, który odpala odpowiednie testy.


Oprócz tego, poszczególne kawałki kodu można testować ręcznie, pisząc odpowiednie unity w katalogu /test i dodając odpowiednie linijki do głównego Makefile'a.
