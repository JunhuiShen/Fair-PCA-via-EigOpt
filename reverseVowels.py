# Given a string s, reverse only all the vowels in the string and return it.

# The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower and upper cases, more than once.

 

# Example 1:

# Input: s = "hello"
# Output: "holle"
# Example 2:

# Input: s = "leetcode"
# Output: "leotcede"
 

# Constraints:

# 1 <= s.length <= 3 * 105
# s consist of printable ASCII characters.

class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowel = ["a","e","i","o","u","A","E","I","O","U"]
        newlist = []
        for element in s:
            if element in vowel:
                newlist.append(element)
        
        newlist.reverse()

        result = list(s)
        newlist_index = 0

        for i in range(len(result)):
            if result[i] in vowel:
                result[i] = newlist[newlist_index]
                newlist_index += 1
        
        return "".join(result)
        

        