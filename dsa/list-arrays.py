strs = ["flow", "flower", "flight"]
first_word = []

def longestCommonPrefix(strs):
  for i in range(len(strs)):
    first_word.append(strs[i])
  return first_word

print(longestCommonPrefix(strs))
  