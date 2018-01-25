# File:  solve_tsp_3.py
#
# This script will solve a TSP via 3 different methods:  nearest neighbor (NN) heuristic, IP, or simulated annealing (SA)
#
# Inputs:
# 	locationsFolder:	For example, practice_3
#	objectiveType:		1 --> Minimize Distance, 2 --> Minimize Time
#	solveNN				1 --> Solve using NN
#	solveIP				1 --> Solve using IP
#	solveSA				1 --> Solve using SA
# 	IPcutoffTime:		-1 --> No time limit, o.w., max number of seconds for Gurobi to run
# 	turnByTurn:			1 --> Use MapQuest for detailed routes. 0 --> Just draw straight lines between nodes.
#
# How to run:
# 	python solve_tsp_3.py practice_3 1 1 1 1 120 1

import sys			# Allows us to capture command line arguments
import csv
import folium	                 # https://github.com/python-visualization/folium
import math
import time
import matplotlib.pyplot as plt


import urllib2
import json
import pandas as pd
from pandas.io.json import json_normalize
import random
from random import randint

from collections import defaultdict

from gurobipy import *


# -----------------------------------------
mapquestKey = '3CAnwO0jyDj9BoE0CTWhLxDZEl2h1Gbw'			# Visit https://developer.mapquest.com/ to get a free API key
#print "YOU MUST SET YOUR MAPQUEST KEY"



# Put your SA parameters here:
#	Tzero:				Initial temperature for SA
#	I:					Number of iterations per temperature for SA
#	delta:				Cooling schedule temp reduction for SA
#	Tfinal:				Minimum allowable temp for SA
#	SAcutoffTime:		Number of seconds to allow your SA heuristic to run
#Tzero = 100
I = 200
#delta = 0.2
#Tfinal = 20
#SAcutoffTime = 20
# -----------------------------------------

# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python
def make_dict():
	return defaultdict(make_dict)

class make_node:
	def __init__(self, nodeName, isDepot, latDeg, lonDeg, demand):
		# Set node[nodeID]
		self.nodeName 	= nodeName
		self.isDepot	= isDepot
		self.latDeg		= latDeg
		self.lonDeg		= lonDeg
		self.demand		= demand

def genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'
	# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)
	routeTypeStr = 'routeType:%s' % transportMode
	
	# Assemble query URL
	myUrl = 'http://www.mapquestapi.com/directions/v2/routematrix?'
	myUrl += 'key={}'.format(mapquestKey)
	myUrl += '&inFormat=json&json={locations:['
	
	# Insert coordinates into the query:
	n = len(coordList)
	for i in range(0,n):
		if i != n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}},'.format(coordList[i][0], coordList[i][1])
		elif i == n-1:
			myUrl += '{{latLng:{{lat:{},lng:{}}}}}'.format(coordList[i][0], coordList[i][1])
	myUrl += '],options:{{{},{},{},{},doReverseGeocode:false}}}}'.format(routeTypeStr, all2allStr,one2manyStr,many2oneStr)
	
	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data

	# This info is hard to read.  Let's store it in a pandas dataframe.
	# We're goint to create one dataframe containing distance information:
	distance_df = json_normalize(data, "distance")
	# print "\nHere's our 'distance' dataframe:"
	# print distance_df	

	# print "\nHere's the distance between the first and second locations:"
	# print distance_df.iat[0,1]	
	
	# Our dataframe is a nice table, but we'd like the row names (indexes)and column names to match our location IDs.
	# This would be important if our locationIDs are [1, 2, 3, ...] instead of [0, 1, 2, 3, ...]
	distance_df.index = locIDlist
	distance_df.columns = locIDlist
		
	# Now, we can find the distance between location IDs 1 and 2 as:
	# print "\nHere's the distance between locationID 1 and locationID 2:"
	# print distance_df.loc[1,2]
		
	
	# We can create another dataframe containing the "time" information:
	time_df = json_normalize(data, "time")

	# print "\nHere's our 'time' dataframe:"
	# print time_df
	
	# Use our locationIDs as row/column names:
	time_df.index = locIDlist
	time_df.columns = locIDlist

	
	# We could also create a dataframe for the "locations" information (although we don't need this for our problem):
	#print "\nFinally, here's a dataframe for 'locations':"
	#df3 = json_normalize(data, "locations")
	#print df3
	
	return(distance_df, time_df)
	

def genShapepoints(startCoords, endCoords):
	# We'll use MapQuest to calculate.
	transportMode = 'fastest'		# Other option includes:  'pedestrian' (apparently 'bicycle' has been removed from the API)

	# assemble query URL 
	myUrl = 'http://www.mapquestapi.com/directions/v2/route?key={}&routeType={}&from={}&to={}'.format(mapquestKey, transportMode, startCoords, endCoords)
	myUrl += '&doReverseGeocode=false&fullShape=true'

	# print "\nThis is the URL we created.  You can copy/paste it into a Web browser to see the result:"
	# print myUrl
	
	# Now, we'll let Python go to mapquest and request the distance matrix data:
	request = urllib2.Request(myUrl)
	response = urllib2.urlopen(request)	
	data = json.loads(response.read())
	
	# print "\nHere's what MapQuest is giving us:"
	# print data
		
	# retrieve info for each leg: start location, length, and time duration
	lats = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lat'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	lngs = [data['route']['legs'][0]['maneuvers'][i]['startPoint']['lng'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	secs = [data['route']['legs'][0]['maneuvers'][i]['time'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]
	dist = [data['route']['legs'][0]['maneuvers'][i]['distance'] for i in range(0,len(data['route']['legs'][0]['maneuvers']))]

	# print "\nHere are all of the lat coordinates:"
	# print lats

	# create list of dictionaries (one dictionary per leg) with the following keys: "waypoint", "time", "distance"
	legs = [dict(waypoint = (lats[i],lngs[i]), time = secs[i], distance = dist[i]) for i in range(0,len(lats))]

	# create list of waypoints (waypoints define legs)
	wayPoints = [legs[i]['waypoint'] for i in range(0,len(legs))]
	# print wayPoints

	# get shape points (each leg has multiple shapepoints)
	shapePts = [tuple(data['route']['shape']['shapePoints'][i:i+2]) for i in range(0,len(data['route']['shape']['shapePoints']),2)]
	# print shapePts
					
	return shapePts


# Capture command line inputs:
if (len(sys.argv) == 8):
	locationsFolder		= str(sys.argv[1])		# Ex:  practice_3
	objectiveType		= int(sys.argv[2])		# 1 --> Minimize Distance, 2 --> Minimize Time
	solveNN				= int(sys.argv[3])		# 1 --> Solve NN.  0 --> Don't solve NN
	solveIP				= int(sys.argv[4])		# 1 --> Solve IP.  0 --> Don't solve IP
	solveSA				= int(sys.argv[5])		# 1 --> Solve SA.  0 --> Don't solve SA	
	IPcutoffTime		= float(sys.argv[6])	# -1 --> No time limit, o.w., max number of seconds for Gurobi to run
	turnByTurn			= int(sys.argv[7])		# 1 --> Use MapQuest for detailed routes.  0 --> Just draw straight lines connecting nodes.
	if (objectiveType not in [1,2]):
		print 'ERROR:  objectiveType %d is not recognized.' % (objectiveType)
		print 'Valid numeric options are:  1 (minimize distance) or 2 (minimize time)'
		quit()
else:
	print 'ERROR: You passed', len(sys.argv)-1, 'input parameters.'
	quit()


# Initialize a dictionary for storing all of our locations (nodes in the network):
node = {}


# Read location data
locationsFile = 'Problems/%s/tbl_locations.csv' % locationsFolder
# Read problem data from .csv file
# NOTE:  We are assuming that the .csv file has a pre-specified format.
#	 Column 0 -- nodeID
# 	 Column 1 -- nodeName
#	 Column 2 -- isDepot (1 --> This node is a depot, 0 --> This node is a customer
#	 Column 3 -- lat [degrees]
#	 Column 4 -- lon [degrees]
#	 Column 5 -- Customer demand
with open(locationsFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
		if (row[0][0] != '%'):
			nodeID = int(row[0])
			nodeName = str(row[1])
			isDepot = int(row[2])
			latDeg = float(row[3])
			lonDeg = float(row[4])
			demand = float(row[5])
	
			node[nodeID] = make_node(nodeName, isDepot, latDeg, lonDeg, demand)

# Use MapQuest to generate two pandas dataframes.
# One dataframe will contain a matrix of travel distances, 
# the other will contain a matrix of travel times.
coordList = []
locIDlist = []
for i in node:
	coordList.append([node[i].latDeg, node[i].lonDeg])
	locIDlist.append(i)

all2allStr	= 'allToAll:true' 
one2manyStr	= 'oneToMany:false'
many2oneStr	= 'manyToOne:false'

[distance_df, time_df] = genTravelMatrix(coordList, locIDlist, all2allStr, one2manyStr, many2oneStr)

# Now, solve the TSP:
[nn_route, ip_route, nn_cost, ip_cost] = [[], [], -1, -1]


def solve_tsp_nn(objectiveType, distance_df, time_df, node):
	# We're going to find a route and a cost for this route
	nn_route = []
	
	# Create a list of locations we haven't visited yet, 
	# excluding the depot (which is our starting/ending location)
	unvisitedLocations = []
	startNode = -1
	for nodeID in node:
		if (node[nodeID].isDepot):
			startNode = nodeID
		else:	
			unvisitedLocations.append(nodeID) 

	if (startNode == -1):
		# We didn't find a depot in our list of locations.
		# We'll let our starting node be the first node
		startNode = min(node)
		print "I couldn't find a depot.  Starting the tour at locationID %d" % (startNode)

		
	# From our starting location, find the nearest unvisited location.
	# We'll also keep track of the total distance (or time)
	i = startNode		# Here's where our tour starts
	nn_route.append(i)
	nn_cost = 0.0
	
#   while len(unvisitedLocations) > 0:
#        # Find the nearest neighbor to customer i
#        if (objectiveType == 1):
#            # Find the location with the minimum DISTANCE to location 1
#            # (excluding any visited locations).
#            # Break ties by choosing the location with the smallest ID.
#            if set(nn_route).issubset(distance_df.columns):
#                test_df = pd.DataFrame(distance_df.drop(distance_df.index[nn_route], axis=1))
#            closestNeighbor = \
#                test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].index.values[0]
#            cost2neighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].values[0]
#        else:
#            # Find the location with the minimum TIME to location 1
#            # (excluding any visited locations).
#            # Break ties by choosing the location with the smallest ID.
#            # print "Unvisited:", unvisitedLocations
#            if set(nn_route).issubset(time_df.columns):
#                test_df = pd.DataFrame(time_df.drop(time_df.index[nn_route], axis=1))
#            closestNeighbor = \
#                test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].index.values[0]
#            cost2neighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].values[0]
#            # print "CLOSEST NEIGHBOUR:", closestNeighbor
#            # print "Cost: ", cost2neighbor
	while len(unvisitedLocations) > 0:
		
		
        
        # Find the nearest neighbor to customer i
		if (objectiveType == 1):
            # Find the location with the minimum DISTANCE to location 1
            # (excluding any visited locations).
            # Break ties by choosing the location with the smallest ID.
			if set(nn_route).issubset(distance_df.columns):
            
				test_df = pd.DataFrame(distance_df.drop(distance_df.index[nn_route], axis=1))
			closestNeighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].index.values[0]
			cost2neighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].values[0]
		else:
            
            # Find the location with the minimum TIME to location 1
            # (excluding any visited locations).
            # Break ties by choosing the location with the smallest ID.
            # print "Unvisited:", unvisitedLocations
			if set(nn_route).issubset(time_df.columns):
				test_df = pd.DataFrame(time_df.drop(time_df.index[nn_route], axis=1))
			closestNeighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].index.values[0]
			cost2neighbor = test_df.loc[i][test_df.loc[i, :] == test_df.loc[i, unvisitedLocations].min()].values[0]
            # print "CLOSEST NEIGHBOUR:", closestNeighbor
            # print "Cost: ", cost2neighbor

		# Add this neighbor to our tour:
		nn_route.append(closestNeighbor)
		
		# Update our tour cost thus far:
		nn_cost += cost2neighbor
		
		# Remove the closestNeighbor from our list of unvisited locations:
		unvisitedLocations.remove(closestNeighbor)
		
		# Virtually move our salesperson to the neighbor we just found:
		i = closestNeighbor
		
	# Finish the tour by returning to the start location:
	nn_route.append(startNode)	
	if (objectiveType == 1):   
		nn_cost += distance_df.loc[i,startNode]
	else:
		nn_cost += time_df.loc[i,startNode]
							
	return (nn_route, nn_cost)

def subtourfun (x,nn1,count):
    if(count == 1):
        [nn , nnkacost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
    else:
        nn = nn1[:]

    #print len(nn)
    multsubtour=[]
    while(len(multsubtour)<x):
        r = randint(1,len(nn)-3)
        a=[]
        b=[]
        #print "r is",r
        for i in range(r,len(nn)-2):
            a.append(i)
        for i in range(r+1,len(nn)-1):# len(nn)-1 can take 0 as well.ie the last entry in the list.
            b.append(i)
        #print "b is ",b
        #print "a is",a
        inda= random.choice(a)
        #print "Random value from list a",inda
        bdash=[]
        temp1 = []
    
        for i in range((inda)+1,len(nn)-1):
            bdash.append(i)
        #print "updated b is",bdash
        indb = random.choice(bdash)
    
        #print "indb is ",indb
        #temp1.append(inda)
        #temp1.append(indb)
        subtour = []
        for i in range(inda,(indb)+1):
            subtour.append(i)
        #print "this is a subtour",subtour
        subtourr =[]
        for i in range(0,len(subtour)):
            temp = subtour[i]
            subtourr.append(nn[temp])
        #print subtourr
    
        revsubtour =[]
        revsubtour = subtourr[::-1]
        #print revsubtour
        temp1.append(inda)
        temp1.append(indb)
        #print "temp 1 value is",temp1
        i=0
        i=i+1
        #print " print kar lavde",i
    
        if(temp1 in multsubtour):
            continue
        else:
            multsubtour.append(temp1)
        #print"multsubtour isko bolte hai",multsubtour
        #print "i is",i
    multsubtourr = []
    a = []
    b = []
    for i in range(0,len(multsubtour)):
        a.append(multsubtour[i][0])
        b.append(multsubtour[i][1])
    for j in range(0, len(multsubtour)):
        temp1 =[]
        for i in range(a[j],b[j]+1):
            temp1.append(nn[i])
        multsubtourr.append(temp1)
    #print "Actual values",multsubtourr
    #print deep_reverse(multsubtourr)
    
    
    multsubtourrev=[]
    #multsubtourrev = reverse(multsubtourr,2)
    
    for j in range(0,len(multsubtourr)):
        multsubtourrev.append(multsubtourr[j][::-1])
    
    
    # for j in range(0,len(multsubtourr)):
    #     multsubtourrev[j] = multsubtourr[j][::-1]
    
    #print "reversed values",multsubtourrev
    
    mylist = []
    
    for j in range(0, len(multsubtourrev)):
        
        list = []
        for i in range(0, a[j]):
            list.append(nn[i])
        for i in range(0, len(multsubtourrev[j])):
            list.append(multsubtourrev[j][i])
        for i in range(b[j]+1, len(nn)):
            list.append(nn[i])
        
        mylist.append(list)
        
        
        
        
        
    #print "HERE IS OUR REVERSED SUBTOUR",mylist  
    return mylist  
[nn , nnkacost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
count =1 
for i in range(0,I+1):
    listoflist = subtourfun(5, nn, count)   
#print "THE VALUE RETURNED FROM THE SUBTOUR FUNCTION IS :",listoflist
#print "THE LENGTH OF LISTOF LIST IS:",len(listoflist)

def costcalculation(objectiveType, haha, distance_df, time_df):
    for i in range(0,len(haha)):
        if (objectiveType == 1):
            cost = 0
            for i in range(0, len(haha)-1):
                j = i+1
                cost += distance_df.loc[haha[i],haha[j]]
        else:
            cost = 0
            for i in range(0, len(haha)-1):
                j = i+1
                cost += time_df.loc[haha[i], haha[j]]

    return cost

cost = {}
#for k in range(0,len(listoflist)-1):
for i in range(0,len(listoflist)):
    cost[i] = costcalculation(objectiveType, listoflist[i], distance_df, time_df)
   
#print "The cost values obtained are as follows:",cost

lowestcostlist=[]
lowestroutelist=[]
def mincostreturn(cost):
    lowest = min(cost, key=cost.get)
    #print " The key corresponding to the lowest cost is",lowest
    
    lowestcost = min(cost.itervalues())
    
    lowestroute = listoflist[lowest]
    #print "lowest cost is :" ,lowestcost
    #print " lowestroute is: " ,lowestroute
    
    return lowestcost, lowestroute
#    for i in range (0,I+1):
#        lowestcostlist.append(lowestcost)
#        lowestroutelist.append(lowestroute)
#        
#    index = np.argmin(lowestcost)
#    LOWCOST = lowestcost[index]
#    LOWROUTE = lowestroute[index]
#    return LOWCOST, LOWROUTE

lowestcostlist, lowestroutelist  = mincostreturn(cost)



def solve_tsp_ip(objectiveType, distance_df, time_df, node, cutoffTime):
   
	ip_route = []
	
	N = []
	q = 0
	for nodeID in node:
		N.append(nodeID)
		if (node[nodeID].isDepot == 0):
			q += 1

	c = defaultdict(make_dict)
	decvarx = defaultdict(make_dict)
	decvaru = {}
	

	# GUROBI
	m = Model("TSP")

	# The objective is to minimize the total travel distance.
	m.modelSense = GRB.MINIMIZE
		
	# Give Gurobi a time limit:
	if (cutoffTime > 0):
		m.params.TimeLimit = cutoffTime
	
	# Define our decision variables (with their objective function coefficients:			
	for i in N:
		decvaru[i] = m.addVar(lb=1, ub=q+2, obj=0, vtype=GRB.CONTINUOUS, name="u.%d" % (i))
		for j in N:
			if (i != j):
				if (objectiveType == 1):
					# We want to minimize distance
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=distance_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))
				else:
					# We want to minimize time
					decvarx[i][j] = m.addVar(lb=0, ub=1, obj=time_df.loc[i,j], vtype=GRB.BINARY, name="x.%d.%d" % (i,j))

	# Update model to integrate new variables:
	m.update()
	
	# Build our constraints:
	# Constraint (2)
	for i in N:
		m.addConstr(quicksum(decvarx[i][j] for j in N if j != i) == 1, "Constr.2.%d" % i)
		
	# Constraint (3)
	for j in N:
		m.addConstr(quicksum(decvarx[i][j] for i in N if i != j) == 1, "Constr.3.%d" % j)
	
	# Constraint (4)
	for i in range(1, q+1):
		for j in N:
			if (j != i):
				m.addConstr(decvaru[i] - decvaru[j] + 1 <= (q + 1)*(1 - decvarx[i][j]), "Constr.4.%d.%d" % (i,j))
				
	# Solve
	m.optimize()


	if (m.Status == GRB.INFEASIBLE):
		# NO FEASIBLE SOLUTION EXISTS
		
		print "Sorry, Guroby is telling us this problem is infeasible."
		
		ip_cost = -999	# Infeasible
	
	elif ((m.Status == GRB.TIME_LIMIT) and (m.objVal > 1e30)):
		# NO FEASIBLE SOLUTION WAS FOUND (maybe one exists, but we ran out of time)

		print "Guroby can't find a feasible solution.  It's possible but one exists."
		
		ip_cost = -888	# Possibly feasible, but no feasible solution found.
			
	else:
		# We found a feasible solution
		if (m.objVal == m.ObjBound):
			print "Hooray...we found an optimal solution."
			print "\tOur objective function value:  %f" % (m.objVal)
		else:
			print "Good News:  We found a feasible solution."
			print "Bad News:  It's not provably optimal."
			print "\tOur objective function value:  %f" % (m.objVal)
			print "\tGurobi's best bound: %f" % (m.ObjBound)

		if (m.Status == GRB.TIME_LIMIT):
			print "\tGurobi reached it's time limit"


		# Let's use the values of the x decision variables to create a tour:
		startNode = -1
		for nodeID in node:
			if (node[nodeID].isDepot):
				startNode = nodeID
	
		if (startNode == -1):
			# We didn't find a depot in our list of locations.
			# We'll let our starting node be the first node
			startNode = min(node)
			print "I couldn't find a depot.  Starting the tour at locationID %d" % (startNode)

		allDone = False
		i = startNode
		ip_route.append(i)
		while (not allDone):
			for j in N:
				if (i != j):
					if (decvarx[i][j].x > 0.9):
						# We traveled from i to j
						ip_route.append(j)
						i = j
						break	# We found the link from i to j.  Stop the "for" loop
			if (len(ip_route) == len(N)):
				# We found all the customers
				allDone = True
		
		# Now, add a link back to the start
		ip_route.append(startNode)
		
		ip_cost = m.objVal
	
	return (ip_route, ip_cost)




plot1 ={}
def solve_tsp_sa():
    Tzero = 500
    I = 1000
    delta = 2
    Tfinal = 10
    SAcutoffTime = 20

    start = time.time()
    
    [Xini, Zini]= solve_tsp_nn(objectiveType, distance_df, time_df, node)
    Xcur = Xini[:] 
    Zcur = Zini
    #count  = 0
    Tcur = Tzero
    Xbest = Xcur[:]
    Zbest = Zcur
    
    
    Zcount = 0
    Xcount = []
    i=0
    count  = 1   
    while (i < I):
        #count = count + 1
        
        #Zcur = costcalculation(objectiveType, Xcur, distance_df, time_df)
        
        #print" bahar wala zcur",Zcur
        listoflist = subtourfun(5,Xcur,count)
        cost = {}
        for i in range(0,len(listoflist)):
            cost[i] = costcalculation(objectiveType, listoflist[i], distance_df, time_df)
        lowestcost, lowestroute  = mincostreturn(cost)
        Zcount = lowestcost
        
        Xcount = lowestroute[:]
        #for i in range(1, I +1):
    
        
        
        #print " hmmmmmmmmmmmmmmmmmmmmmmmmmmmm" , Zcount
        #print " hmmmmmmmmmmmmmmmmmmmmmmmmmmmm" , Xcount
        if (Zcount < Zcur):
            #Xcur = list(Xcount)
            Xcur=Xcount[:]
            Zcur = (Zcount)
            plot1[count]=Zcur
            #plot1.update({count : Zcur})
            #print "Now Xcount is ", Xcount
            #print " Now Zcount is " , Zcount
            plott1, = plt.plot(count,Zcur,'rD',ms = 12) # ALWAYS Accepted values
            
        else:
            deltaC =  Zcur - Zcount
            #print " zcurrent is",Zcur
            #print "zcount is",Zcount
            #print "deltaC #######################################",deltaC
            if((random.uniform(0,1)) <= 2.7182818**(deltaC/ Tcur)):
                Xcur= Xcount
                Zcur = Zcount
                plot1[count]=Zcur
                #plt.plot(i, Zcount, 'H-',ms =8,color ='blue')#worst but accepted
                plott2, = plt.plot(count, Zcur, 'gH',ms =12)
                #print (math.exp((deltaC)/float(Tcur)))
                print "TEMP YE HAI " , Tcur
                #print " Xcount in if loop is:" ,Xcur
                print " Zcount in if loop is:" ,Zcur
            else:
                plott3, = plt.plot(count, Zcount, 'H',ms =12,color ='yellow')#NOT ACCEPTED
                
                
            #print deltaC
        if(Zcur< Zbest):    
            Zbest = Zcur
            Xbest = Xcur[:]
            #plt.plot(count, Zbest, 'H-',ms =8,color ='yellow')
        Tcur = Tcur - delta
        i = i +1
        count = count +1
        end = time.time()
        if (Tcur < Tfinal):
            break
        elif((start -end) >= SAcutoffTime):
            break
        else:
            continue
            
    print "Zbest is ", Zbest
    print " Xbest is ", Xbest
    lists = sorted(plot1.items())
    x,y =zip(*lists)
    plt.plot(x,y)
    plt.title('The variation of objective function vs no of iterations')
    plt.ylabel('objective_function_value')
    plt.xlabel('No of iterations')
    plt.legend([plott1,plott2,plott3],["good values,always accepted","bad values,still accepted","bad values,rejected"])
    plt.show()
      
    return (Xbest , Zbest)
    
	
	# INSERT
	#
	# YOUR
	#
	# SIMULATED ANNEALING 
	#
	# CODE
	# 
	# HERE
	
	#return (sa_route, sa_cost)

	

if (solveNN):
	# Solve the TSP using nearest neighbor
	[nn_route, nn_cost] = solve_tsp_nn(objectiveType, distance_df, time_df, node)
if (solveIP):
	# Solve the TSP using the IP model
	[ip_route, ip_cost] = solve_tsp_ip(objectiveType, distance_df, time_df, node, IPcutoffTime)
if (solveSA):
	# Solve the TSP using simulated annealing
	[sa_route, sa_cost] = solve_tsp_sa()		# FIXME -- YOU'LL NEED TO PASS SOME INFO TO THIS FUNCTION

	
 

# Create a map of our solution
mapFile = 'Problems/%s/osm.html' % locationsFolder
map_osm = folium.Map(location=[node[0].latDeg, node[0].lonDeg], zoom_start=10)



# Plot markers
for nodeID in node:
	if (node[nodeID].isDepot):		
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='red'), popup = node[nodeID].nodeName).add_to(map_osm)
	else:
		folium.Marker([node[nodeID].latDeg, node[nodeID].lonDeg], icon = folium.Icon(color ='blue'), popup = node[nodeID].nodeName).add_to(map_osm)
	
if (turnByTurn):
	# (PRETTY COOL) Plot turn-by-turn routes using MapQuest shapepoints:
	if (nn_cost > 0):
		# a) nearest neighbor:
		i = nn_route[0]
		for j in nn_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="red", weight=10, opacity=0.5).add_to(map_osm)	
	
			i = j
			
	if (ip_cost > 0):
		# b) ip:
		i = ip_route[0]
		for j in ip_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="green", weight=4.5, opacity=0.5).add_to(map_osm)	
	
			i = j
            
	if (sa_cost > 0):
		# b) ip:
		i = sa_route[0]
		for j in sa_route[1:]:
			startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
			endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
			
			myShapepoints = genShapepoints(startCoords, endCoords)	       
		
			folium.PolyLine(myShapepoints, color="blue", weight=5, opacity=0.5).add_to(map_osm)	
	
			i = j

    
#	if (sa_cost > 0):
#        i = sa_route[0]
#        for j in sa_route[1:]:
#            startCoords = '%f,%f' % (node[i].latDeg, node[i].lonDeg)
#            endCoords = '%f,%f' % (node[j].latDeg, node[j].lonDeg)
#            myShapepoints = genShapepoints(startCoords, endCoords)
#            folium.PolyLine(myShapepoints, color="blue", weight=8, opacity=0.5).add_to(map_osm)	
#            i = j
#            
            

            
            
            
    
            
            
else:
	# (BORING) Plot polylines connecting nodes with simple straight lines:
	if (nn_cost > 0):
		# a) nearest neighbor:
		points = []
		for nodeID in nn_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="red", weight=8.5, opacity=0.5).add_to(map_osm)	
	if (ip_cost > 0):
		# b) ip:
		points = []
		for nodeID in ip_route:
			points.append(tuple([node[nodeID].latDeg, node[nodeID].lonDeg]))
		folium.PolyLine(points, color="green", weight=4.5, opacity=0.5).add_to(map_osm)	

map_osm.save(mapFile)

print "\nThe OSM map is saved in: %s" % (mapFile)

if (solveNN):
	print "\nNearest Neighbor Route:"
	print nn_route
	print "Nearest Neighbor 'Cost':"
	print nn_cost
if (solveIP):	
	print "\nIP Route:"
	print ip_route
	print "IP 'cost':"
	print ip_cost
if (solveSA):	
	print "\nSA Route:"
	print sa_route
	print "SA 'cost':"
	print sa_cost









































