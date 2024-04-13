## Lets Write a Story: Sequence Models for Text Generation &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KO4JeIHRiKxiRJdK7gY-B9bZGfDSvCt_?usp=sharing)

The ability to write convincing text has always been a human ability. This was the intial purpose
of the Turing test to see if humans could tell the difference between machine generated text and 
human written ones. Todays Large Language Models (LLM) have far surpassed the turing test, but they are
fundamentally very similar to this dummy LSTM example. 

Our goal is to train a Next Character prediction LSTM model to generate new text. 


```
------------------------------------
Epoch 0
Loss 4.514334201812744
Sample Generation
Spells i8yE",t&(lF/TwSMJ|M~9ktGz■cd2V1a;IMd>f;/P!?"P90|wgt1Nzb)•w|E"V"‘S-Y
.>mf1Q?%.'%LOZw□u>M‘S•(/x,v•)n16xe?\?GcG~a3u”GX~U9j?;3thH%M,8Kk•-6YlkG1:?/1
“buVAY□FmM2jf/U’:‘o•J|Eore3vvNDI%0sb%c4C(.dN|;~.O7“0G5mS—!
------------------------------------
------------------------------------
Epoch 2700
Loss 1.0395761728286743
Sample Generation:
Spells of pain id lain she arrived, now. They had no longer and Harry’s letter. 
Everyone was beetless and Hermione were uncoving him in neck. He less that no 
one seemed, “Pcoragos, used. I don’t.” said Mr. M
------------------------------------
```

We can see that in about 2700 iterations, we have been able to reach text that looks like English, each word is close to English, but there 
is no real idea conveyed and it does't make sense as a sentence. This is a problem with character level models
so to get more convinving performance we would have to use a proper tokenizer. I again wanted to avoid this as we want to 
make this as user-friendly as possible!

In this notebook you will cover: 
- Passing in a sequence vs iterating over a sequence and passing a token at at time to the LSTM
- Dataset preparation to load text from Harry Potter
- Build PyTorch model that can both train weights and generate text given some starting string!