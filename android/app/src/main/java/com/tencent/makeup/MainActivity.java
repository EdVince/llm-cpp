// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.makeup;
import android.annotation.SuppressLint;
import android.content.Context;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity
{


    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE"
    };

    public void verifyStoragePermission(Activity activity){
        try{
            int permission = ActivityCompat.checkSelfPermission(activity,"android.permission.WRITE_EXTERNAL_STORAGE");
            if(permission!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(activity,PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
            }
        }catch (Exception e){
            e.printStackTrace();
            e.printStackTrace();
        }
    }

    private StableDiffusion sd = new StableDiffusion();

    private Button sendButton;
    private TextView showText;
    private EditText inText;

    /** Called when the activity is first created. */
    @SuppressLint("MissingInflatedId")
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        verifyStoragePermission(this);

        showText = (TextView) findViewById(R.id.showText);
        inText = (EditText) findViewById(R.id.inText);
        sendButton = (Button) findViewById(R.id.sendBtn);

        String path = "/data/data/com.tencent.makeup/files/";
        File directory = new File(path);
        File[] filesAndDirectories = directory.listFiles();
        List<String> folderNameArray = new ArrayList<>();
        if (filesAndDirectories != null) {
            for (File file : filesAndDirectories) {
                if (file.isDirectory()) {
                    folderNameArray.add(file.getName());
                }
            }
        }
        String[] folderName = folderNameArray.toArray(new String[folderNameArray.size()]);

        String target = "Qwen1.5-1.8B";
        for (String folder : folderName) {
            if (folder.startsWith(target)) {
                target = path + folder;
                Log.i("LLM","get model: " + target);
                boolean ret_init = sd.Init(target);
            }
        }

        final String finalTarget = target;
        showText.setText("");
        showText.append("model path: " + finalTarget + "\n\n");

        sendButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                showText.setText("");
                showText.append("model path: " + finalTarget + "\n\n");

                String input = inText.getText().toString();
                showText.append(" user: "+input+"\n\n");

                String output = sd.gen(input);
                showText.append("robot: "+output+"\n");

                inText.getText().clear();
            }
        });

    }
}
