function lmk = ReadLmk(file)

fid = fopen(file,'r');
line = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    if line < 4
        if strcmp(tline(1:8),'n_points')
            num = str2num(tline(10:end));
            lmk = zeros(num,2);
            tline = fgetl(fid); line = line +2;
            continue;
        else
            line = line + 1;
            continue;
        end
    elseif and(line > 3, line <= 3+num)
        lmk(line-3,:) = str2num(tline);
        line = line + 1;
    end
end
fclose(fid);
