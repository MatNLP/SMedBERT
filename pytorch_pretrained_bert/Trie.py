class Trie:
    # word_end = -1
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.word_end = -1

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]

        curNode[self.word_end] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]

        # Doesn't end here
        if self.word_end not in curNode:
            return False

        return True

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curNode = self.root
        if prefix not in curNode: return False
        return True
        #     if not c in curNode:
        #         return False
        #     curNode = curNode[c]
        # return True
    def delete(self,w):
        curNode=self.root
        for c in w:
            if c not in w:
                return False
            curNode=curNode[c]
        if self.word_end not in curNode:
            return False
        curNode.pop(self.word_end)
        return True
    def findApperanceWord(self,sentences):
        s=set()
        hit_word=[]
        hit_pos=[]
        for start in range(len(sentences)):
          hit=''
          c=start
          curNode=self.root
          while(True):
            if c>=len(sentences) or sentences[c] not in curNode:
              break
            else:
              curNode=curNode[sentences[c]]
              hit+=sentences[c]
            if(self.word_end in curNode):
              hit_word.append(hit)
              hit_pos.append(start)
              s.add(hit)
            c+=1
        # print(hit_list)
        # raise ValueError('End')
        return s,hit_word,hit_pos
    def get_lengest_match(self,chars,cand_index):
        curNode=self.root
        last=None
        for k in range(cand_index,len(chars)):
            if chars[k] not in curNode: break
            curNode=curNode[chars[k]]
            if(self.word_end in curNode):
                last=k
        return last

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)