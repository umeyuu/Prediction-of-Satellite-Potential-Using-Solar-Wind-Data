import urllib.request
import os


def save_satelite_data(name):
    for year in range(2010, 2015):
        for month in range(1, 13):
            for day in range(1, 32):
                year = str(year)
                month = str(month).zfill(2)
                day = str(day).zfill(2)
                path = f'https://satdat.ngdc.noaa.gov/dmsp/data/{name}/ssj/{year}/{month}/dmsp-{name}_ssj_precipitating-electrons-ions_'+year+month+day+'_v1.1.2.cdf'
                save_dir = f'./DATA/dmsp-{name}/{year}/{month}/'
                file_name = f'dmsp-{name}_{year}{month}{day}.cdf' 

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                try:
                    urllib.request.urlretrieve(path, save_dir+file_name)
                except:
                    print(f'{file_name}の保存に失敗しました。')
                    continue


def serch_fail_to_save(name):
    files = []
    for year in range(2010, 2015):
        for month in range(1, 13):
            for day in range(1, 32):
                year = str(year)
                month = str(month).zfill(2)
                day = str(day).zfill(2)
                path = f'https://satdat.ngdc.noaa.gov/dmsp/data/{name}/ssj/{year}/{month}/dmsp-{name}_ssj_precipitating-electrons-ions_'+year+month+day+'_v1.1.2.cdf'
                save_dir = f'./DATA/dmsp-{name}/{year}/{month}/'
                file_name = f'dmsp-{name}_{year}{month}{day}.cdf' 

                if not os.path.isfile(save_dir+file_name):
                    files.append((path, save_dir+file_name))
    for path, save_path in files:
        try:
            urllib.request.urlretrieve(path, save_path)
        except:
            print(f'{save_path}の保存に失敗しました。')
            continue

def check_save_term(name):
    count = 0
    for year in range(2010, 2015):
        for month in range(1, 13):
            for day in range(1, 32):
                if month in [4, 6, 9, 11] and day==31:
                    continue

                year = str(year)
                month = str(month).zfill(2)
                day = str(day).zfill(2)
                save_dir = f'./DATA/dmsp-{name}/{year}/{month}/'
                file_name = f'dmsp-{name}_{year}{month}{day}.cdf' 

                if os.path.isfile(save_dir+file_name):
                    if count == 0:
                        st = f'{year}/{month}/{day}'
                    count += 1
                    et = f'{year}/{month}/{day}'
                else:
                    if count == 0:
                        continue
                    print(f'{st}~{et}まで{count}個あるよ。')
                    count = 0

save_satelite_data('f18')
# check_save_term('f17')
# breakpoint()