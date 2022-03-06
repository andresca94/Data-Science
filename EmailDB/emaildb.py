import sqlite3
import re

conn = sqlite3.connect('emaildb1.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts')

cur.execute('''
CREATE TABLE Counts (org TEXT, count INTEGER)''')

fname = input('Enter file name: ')
if (len(fname) < 1): fname = 'mbox-short.txt'
fh = open(fname)
for line in fh:
    if not line.startswith('From: '): continue
    pieces = line.split()
    email = pieces[1]
    org = re.findall('.+@(.+)',email)
    org = org[0]
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (org,))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count)
                VALUES (?, 1)''', (org,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE org = ?',
                    (org,))
    conn.commit()


sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

org = list()
count = list()
for row in cur.execute(sqlstr):
    #print(str(row[0]), row[1])
    org.append(row[0])
    count.append(row[1])

cur.close()

conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts')

cur.execute('''
CREATE TABLE Counts (org TEXT, count INTEGER)''')

for i in range(len(org)):
    cur.execute('INSERT INTO Counts (org, count) VALUES (?, ?)',(org[i],count[i]))
    conn.commit()

cur.close()
