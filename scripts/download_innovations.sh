#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Function to log messages
log() {
    echo "$1"
}

get_latest_dirs() {
    dirs=$(ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME "moo ls moose:/adhoc/projects/nwsclimatedatasharing/PML_AMM15/cycle/ | sort | tail -n 2")

    # Check if the SSH command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to execute 'moo ls' command." >&2
        exit 1
    fi

    latest_dir=$(echo "$dirs" | tail -n 1)
    second_latest_dir=$(echo "$dirs" | head -n 1)
    echo "${latest_dir: -8} ${second_latest_dir: -8}"
}

get_all_file_types() {
    echo "increments model_errors"
}

show_help() {
    echo "Usage: $0 [-d model_date] [-t file_type] [-p local_path]"
    echo "Options:"
    echo "  -d model_date     Optional. The date to use for the directory (format: YYYYMMDD). Defaults to the latest available date."
    echo "  -t file_type      Optional. The type of files to download. Can be 'increments', 'model_errors', or leave empty for both."
    echo "  -p local_path     Optional. The local path where files will be saved. Defaults to ~/data."
    echo "  -h                Display this help message."
}

MODEL_DATE=""
FILE_TYPE=""
LOCAL_PATH="$HOME/data"

# Parse command-line arguments
while getopts "d:t:p:h" opt; do
    case ${opt} in
        d )
            MODEL_DATE=$OPTARG
            ;;
        t )
            FILE_TYPE=$OPTARG
            ;;
        p )
            LOCAL_PATH=$OPTARG
            ;;
        h )
            show_help
            exit 0
            ;;
        \? )
            show_help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

ssh-add $JASMIN_SSH_FILE

if [ -z "$MODEL_DATE" ]; then
    IFS=' ' read -r LATEST_DIR SECOND_LATEST_DIR <<< $(get_latest_dirs)
    MODEL_DATE=$LATEST_DIR
    log "Using latest directory: $MODEL_DATE"
else
    SECOND_LATEST_DIR=""
fi

# If no file_type is provided, use all file types
if [ -z "$FILE_TYPE" ]; then
    FILE_TYPES=$(get_all_file_types)
    log "Using all file types: $FILE_TYPES"
else
    FILE_TYPES=$FILE_TYPE
fi


download_files() {
    local date=$1
    log "Downloading files from MOOSE for date $date to $MASS_HOSTNAME..."

    ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME << EOF
        mkdir -p ~/output
        cd ~/output
        echo "Downloading files for $date..."
        for file_type in $FILE_TYPES; do        
            if [[ \$file_type == "increments" ]]; then
                moo get -v moose:/adhoc/projects/nwsclimatedatasharing/PML_AMM15/cycle/$date/assim_daym2.amm15 . || echo "Error downloading increments for $date"
            elif [[ \$file_type == "model_errors" ]]; then
                moo get -v moose:/adhoc/projects/nwsclimatedatasharing/PML_AMM15/cycle/$date/fdbk.obsop.daym2.amm15/profb_01.nc . || echo "Error downloading model_errors for $date"
            else 
                echo "Invalid file type: \$file_type"
            fi
        done
        for file in *; do
            mv "\$file" "innovations_${date}_\$file"
        done
EOF
}

download_files $MODEL_DATE

output_files=$(ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME 'ls ~/output/')

if [ -z "$output_files" ] && [ -n "$SECOND_LATEST_DIR" ]; then
    log "No files found for the latest directory. Trying the second latest directory: $SECOND_LATEST_DIR"
    MODEL_DATE=$SECOND_LATEST_DIR
    download_files $MODEL_DATE
    log "Files downloaded to $MASS_HOSTNAME using MOOSE."
    output_files=$(ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME 'ls ~/output/')
fi

if [ -z "$output_files" ]; then
    log "No files found in the output directory. Check the input arguments or change the model_date."
    exit 1
fi

log "Files downloaded to $MASS_HOSTNAME using MOOSE for date $MODEL_DATE."

log "Downloading files from $MASS_HOSTNAME to local machine..."

rsync -avz -e "ssh -o 'ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME' -i $JASMIN_SSH_FILE" $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME:~/output/* $LOCAL_PATH

log "Files downloaded from $MASS_HOSTNAME to local machine."

remote_files=$(ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME 'ls ~/output/')

# Check each remote file in the local directory
for remote_file in $remote_files; do
    if [ ! -f "$LOCAL_PATH/$remote_file" ]; then
        log "File $remote_file was not copied correctly."
    else
        log "File $remote_file copied successfully."
    fi
done

log "Removing files from $MASS_HOSTNAME..."
ssh -o "ProxyJump $JASMIN_LOGIN_USERNAME@$JASMIN_LOGIN_HOSTNAME" -i $JASMIN_SSH_FILE $JASMIN_LOGIN_USERNAME@$MASS_HOSTNAME << EOF
    rm -rf ~/output/
EOF

log "Files removed from $MASS_HOSTNAME."
