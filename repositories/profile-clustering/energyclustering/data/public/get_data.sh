#!/bin/bash
# the master table
wget https://www.fluvius.be/sites/fluvius/files/2019-02/master-table-meters.csv
# the table with energy consumption
wget https://www.fluvius.be/sites/fluvius/files/2019-02/READING_2016.zip
unzip READING_2016.zip
rm READING_2016.zip
