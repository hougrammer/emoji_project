# emoji_project

## Examining the Expressive Power of Emoji
David Hou
Michelle Cutler
Sergio Ferro

### Proposal
We propose exploring the expressive power of emoji by translating English into sequences of emoji and then back into English.  After these two translations, we can measure semantic similarity between input and output, indicating the “expressiveness” of emoji.  This similarity measure can be computed by averaging the embedding vectors for the inputs and outputs and taking the cosine distance (similar to emoji2vec).  For this project we consider several architectures: e.g. twin RNNs, variational autoencoder, or simple n-gram models.  If successful, our project will provide insight on how emoji express universal ideas through pictographs.  This has implications in social media sentiment analysis and cross-lingual communications.

Much of existing NLP emoji literature involves predicting emojis from blocks of text (e.g. Barbieri et al).  We introduce the extra challenge of converting English into longer sequences of emoji.  For example, we aim to translate “David goes to UC Berkeley” to “👨→🏫” and back to “man goes to school”.  Some information, like proper nouns, are expected to be lost in the translations.  Therefore, more common translation evaluations such as BLEU score are not as well-suited as embedding comparison described above.

We will leverage many existing emoji datasets, supplemented with data from Twitter or Instagram.  EmojiNet offers a large repository of detailed descriptions of thousands of emoji. We will train embeddings using this description data following following the work of Wijernati et al .  These embeddings will help us establish a baseline translation model (emoji have no strict grammar or syntax, we may get a decent translation simply by comparing embeddings of English words and emoji).  For a large example of emoji translation, see the Emoji Dick reference, which utilized Amazon Mechanical Turk to translate all of Moby Dick into emoji.

### References / Notes
https://arxiv.org/pdf/1609.08359.pdf emoji2vec
http://emojinet.knoesis.org/home.php emojinet
https://arxiv.org/pdf/1707.04653.pdf Semantics based measure of emoji similarity
http://www.aclweb.org/anthology/E17-2017 Are emojis predictable?
http://www.aclweb.org/anthology/U16-1018 EDA of emoji translation of Moby Dick
http://www.czyborra.com/unicode/emojidick.pdf PDF version of Emoji Dick
http://aclweb.org/anthology/N18-2107 Multimodal emoji prediction (text + pictures)
http://aclweb.org/anthology/S18-1003 Emoji prediction competition in English and Spanish
https://www.aclweb.org/anthology/D14-1179 RNN encoder / decoder
arXiv:1606.05908 Tutorial in Variational Autoencoders
