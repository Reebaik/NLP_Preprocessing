import re  # re = regular expression module used for pattern matching

# 🔹 Sample multi-line input text containing emails (some valid, some invalid)
text = """
Hello, please contact us at nibinjoseph@gmail.com for support.
You can also reach out to reebaikhamilton@company.co.uk , sreehari@mysite.org, sonatbenny@gmail.com or adityaj@gmail.com.
Invalid emails like test@com or @wrong.com should be ignored.
"""

# 🔍 Email pattern using regex:
# Explanation:
# \b                    = word boundary (to avoid matching inside other words)
# [a-zA-Z0-9._%+-]+     = username part (can include letters, numbers, dots, etc.)
# @                     = mandatory "@" symbol
# [a-zA-Z0-9.-]+        = domain name (can include letters, numbers, hyphens)
# \.                    = dot before domain extension
# [a-zA-Z]{2,}          = domain extension (like com, org, co.uk — must be at least 2 letters)
# \b                    = end word boundary
email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'

# ✅ re.findall() returns a list of all substrings in the text that match the regex
emails = re.findall(email_pattern, text)

# 📋 Print the extracted emails one by one
print("Extracted Emails:")
for email in emails:
    print("-",email)  