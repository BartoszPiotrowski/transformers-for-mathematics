cat $1 | while read l
do
    sed 's/\([0-9]\)/\1 /g; s/ \+/ /g; s/ $//g; s/-/ - /g; s/\*/ \* /g; s/  / /g; s/\* \*/\*\*/g; s/^ //g'
done
