# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
create app fatal event
"""
import json
import os
import sys

import requests


def get_ep():
    """
    get end point
    """
    host = os.environ.get("SYS_API_HOST", "paddlecloud.baidu-int.com")
    port = os.environ.get("SYS_API_PORT", "80")
    return "{}:{}".format(host, port)


def post(path, data, token):
    """
    post data
    """
    ep = get_ep()
    headers = {
        "token": token,
    }
    if not isinstance(data, str):
        data = json.dumps(data)
    url = "http://{}{}".format(ep, path)
    result = requests.post(url, data=data, headers=headers)
    if result.status_code == 200:
        return True, result.text
    else:
        return False, result.text


def create_event(level, title, message):
    """
    create_event
    """
    longjob_id = os.environ.get("PDC_LONGJOB_ID", "")
    token = os.environ.get("PDC_TOKEN", "")
    token = "{}/{}".format(longjob_id, token)
    req = {
        "longjobId": longjob_id,
        "title": title,
        "message": message,
    }
    if level == "error":
        path = "/inner/v3/event/apperror"
    elif level == "notice":
        path = "/inner/v3/event/appnotice"
    else:
        raise Exception("unknown level: {}, only support error and notice".format(level))

    ok, res = post(path, req, token)
    if ok:
        print("create app notice event success")
        sys.exit(0)
    else:
        print("create app notice event failed, body: {}, reason: {}".format(req, res))
        sys.exit(1)


def main():
    """
    main
    """
    if len(sys.argv) < 3:
        print('usage: python create_msg_event.py notice/error DiskPressure "disk is full"')
        exit(1)
    level = sys.argv[1]
    title = sys.argv[2]
    message = sys.argv[3]
    create_event(level, title, message)


if __name__ == "__main__":
    main()
