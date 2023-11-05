import pandas

name = ['Id','age','sex','bmi','children','smoker','region','charges']

dt = pandas.read_csv('pracmodels\medcos.csv', names= name)

dt.describe