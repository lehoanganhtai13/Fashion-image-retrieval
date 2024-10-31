#!/bin/bash

# Start Elasticsearch in the background
bin/elasticsearch &

# Wait for Elasticsearch to start
sleep 30

# Set the password for the elastic user
expect << EOF
spawn bin/elasticsearch-reset-password -u elastic -i
expect "Please confirm that you would like to continue"
send "y\r"
expect "Enter password for \[elastic\]:"
send "${ELASTIC_PASSWORD}\r"
expect "Re-enter password for \[elastic\]:"
send "${ELASTIC_PASSWORD}\r"
expect eof
EOF

# Set the password for the kibana_system user
expect << EOF
spawn bin/elasticsearch-reset-password -u kibana_system -i
expect "Please confirm that you would like to continue"
send "y\r"
expect "Enter password for \[kibana_system\]:"
send "${ELASTIC_PASSWORD}\r"
expect "Re-enter password for \[kibana_system\]:"
send "${ELASTIC_PASSWORD}\r"
expect eof
EOF

# Keep the container running
wait