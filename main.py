# import smtplib

# import requests


# def Dose_Availability_Lon_Lat(Lattitude,Longitude):
#     api="https://cdn-api.co-vin.in/api/v2/appointment/centers/public/findByLatLong?lat={}&long={}".format(Lattitude,Longitude)
#     return main_task(api)

# def Dose_Availability_District(district_id,date):
#     api="https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByDistrict?district_id={}&date={}".format(district_id,date)
#     return main_task(api)

# def Dose_Availability_Pincode(pincode, date):
#     api = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByPin?pincode={}&date={}".format(pincode,date)

#     return main_task(api)

# def main_task(api):
#     response=requests.get(api)
#     data=response.json()['sessions']
#     output="*"*30
#     for area in data:
#         if area['available_capacity']>0:
#             # for (field,value) in area.items():
#             #     print(field,':',value)
#             # else:
#             output+="  Hospital Name:" + area['name'] + "*"*30 +"\n"
#             output+='''\
# Address: {}
# Pincode: {}
# available_capacity_dose1 : {}
# available_capacity_dose2 : {}
# available_capacity : {}
# min_age_limit: {}
# Time Slots: {}

# '''.format(area['address'],area['pincode'],area['available_capacity_dose1'],area['available_capacity_dose2'],
#                              area['available_capacity'],area['min_age_limit'],str(area['slots'])[1:-1])
#             output+="*"*30
#     return output

# def send_email(email,message):
#     host = "smtp.gmail.com"
#     port = 587

#     connection=smtplib.SMTP(host,port)
#     connection.starttls()

#     username="innovateyourself2build@gmail.com"
#     with open("creds.txt") as file:
#         password=file.read()
#     connection.login(username,password)

#     receiver=email
#     subject="Test Email"
#     # message="This is a test email for demo. Ashish Saini"

#     body='''\
# From: {}
# Subject:{}

# {}'''.format(username,subject,message)

#     connection.sendmail(username,receiver,body)
#     connection.quit()

# send_email("innovateyourself2build@gmail.com",'test mail')
# print(Dose_Availability_Pincode(110052,"31-05-2021"))

# main.py
import requests

def Dose_Availability_Pincode(pincode, date):
    api = f"https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByPin?pincode={pincode}&date={date}"
    return main_task(api)

def main_task(api):
    headers = {"User-Agent": "Mozilla/5.0"}  # Required by CoWIN
    response = requests.get(api, headers=headers)

    try:
        data = response.json().get('sessions', [])
    except Exception as e:
        return f"❌ API Error: {e}\nRaw: {response.text[:200]}"

    if not data:
        return "⚠️ No vaccination sessions found for this pincode & date."

    output = "*" * 30 + "\n"
    for area in data:
        if area['available_capacity'] > 0:
            output += (
                f"Hospital Name: {area['name']}\n"
                f"Address: {area['address']}\n"
                f"Pincode: {area['pincode']}\n"
                f"available_capacity_dose1 : {area['available_capacity_dose1']}\n"
                f"available_capacity_dose2 : {area['available_capacity_dose2']}\n"
                f"available_capacity : {area['available_capacity']}\n"
                f"min_age_limit: {area['min_age_limit']}\n"
                f"Time Slots: {', '.join(area['slots'])}\n"
                + "*" * 30 + "\n"
            )

    return output if len(output.strip("*\n")) > 0 else "⚠️ No available slots at the moment."

