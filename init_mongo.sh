#!/bin/bash

# Check if the username and password were passed as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: init-mongo.sh <username> <password>"
  exit 1
fi

MONGO_USERNAME=$1
MONGO_PASSWORD=$2

echo "Creating user in MongoDB..."
# Start MongoDB without authentication to allow user creation
mongod --bind_ip_all --logpath /var/log/mongod.log --fork

# Wait a few seconds for MongoDB to start
sleep 5

# Check if the user already exists
USER_EXISTS=$(mongo --quiet --eval "db.getUser('$MONGO_USERNAME')" admin)

if [ "$USER_EXISTS" == "null" ]; then
  # User doesn't exist, create the user
  echo "Creating user $MONGO_USERNAME"
  mongo --eval "db.createUser({user: '$MONGO_USERNAME', pwd: '$MONGO_PASSWORD', roles:[{role: 'root', db: 'admin'}]})" admin
else
  echo "User $MONGO_USERNAME already exists, skipping user creation"
fi
# Now stop the current mongod process
pkill mongod

# Wait a few seconds for the process
sleep 5

# Restart MongoDB with authentication enabled
mongod --auth --bind_ip_all --logpath /var/log/mongod.log --fork

import_collection () {
    mongoimport --uri mongodb://$MONGO_USERNAME:$MONGO_PASSWORD@localhost:27017/arxivdb?authSource=admin --db arxivdb --collection arxivcol --type json --file /collection.json --jsonArray
}


echo "MongoDB configured successfully. Importing collection."
# Check if the database and collection exist
DB_EXISTS=$(mongo mongodb://$MONGO_USERNAME:$MONGO_PASSWORD@localhost:27017/default?authSource=admin --eval "db.getMongo().getDBNames()")

if echo "$DB_EXISTS" | grep -q "arxivdb"; then
  echo "Database arxivdb already exists, checking if collection exists"
  COL_EXISTS=$(mongo mongodb://$MONGO_USERNAME:$MONGO_PASSWORD@localhost:27017/arxivdb?authSource=admin --eval "db.getCollectionNames()" arxivdb)
  if echo "$COL_EXISTS" | grep -q "arxivcol"; then
    echo "Collection arxivcol already exists in arxivdb, skipping import"
  else
    # Collection doesn't exist, run mongoimport
    echo "Importing collection into arxivdb"
    import_collection
  fi
else
  # DB doesn't exist, run mongoimport
  echo "Importing collection into arxivdb"
  import_collection
fi

sleep 5

echo "Mongo DB setup complete. Waiting for connections..."

tail -f /dev/null
