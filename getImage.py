import sys

sys.path.append(r'../')
import base
from githubJet import github

proxyInfo = base.getProxy()

join = github(
    email = '',
    password = '',
    totpString = '',
    emailPassword = '',
    proxy = '',
    proxy2 = proxyInfo['hostProxy2'],
    school = [],
    userAgent = ''
)
for _ in range(200):
    join.captcha_token()
